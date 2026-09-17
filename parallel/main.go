package main

import (
	"flag"
	"fmt"
	"math"
	"runtime"
	"simd"
	"sync"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const (
	G                   float32 = 1000.0 // gravitational constant
	GRAVITY_SOFTENINING float32 = 5.0
	PHYSICS_DT          float32 = 1.0 / 120.0
	RADIUS_SCALE        float32 = 0.25
	SPAWN_RADIUS        float32 = 350.0
)

var (
	countPtr *int     = flag.Int("count", 1000, "Particle count")
	ringPtr  *int     = flag.Int("ring", 333, "number of concentric circles")
	speedPtr *float64 = flag.Float64("speed", 0.3, "speed of particles")
)

type pool struct {
	numWorkers  int
	starts      []int
	ends        []int
	workSignals []chan struct{}
	wg          sync.WaitGroup
}

func newPool(n, numWorkers int) *pool {
	if numWorkers < 1 {
		numWorkers = 1
	}

	chunk := (n + numWorkers - 1) / numWorkers

	p := &pool{}

	for w := range numWorkers {
		start := w * chunk
		end := start + chunk

		if start >= n {
			break
		}

		if end > n {
			end = n
		}

		p.starts = append(p.starts, start)
		p.ends = append(p.ends, end)
		p.workSignals = append(p.workSignals, make(chan struct{}))
	}
	p.numWorkers = len(p.starts)

	return p
}

func (p *pool) spawn(ps particles, lanes int) {
	for w := range p.numWorkers {
		go func(w int) {
			for range p.workSignals[w] {
				calcAccelerationRange(ps, lanes, p.starts[w], p.ends[w])
				p.wg.Done()
			}
		}(w)
	}
}

func (p *pool) run() {
	p.wg.Add(p.numWorkers)
	for w := range p.numWorkers {
		p.workSignals[w] <- struct{}{}
	}
	p.wg.Wait()
}

func (p *pool) close() {
	for _, ch := range p.workSignals {
		close(ch)
	}
}

type particles struct {
	posX, posY []float32
	velX, velY []float32
	accX, accY []float32
	mass       []float32
}

func allocParticles(count int) particles {
	return particles{
		posX: make([]float32, 0, count),
		posY: make([]float32, 0, count),
		velX: make([]float32, 0, count),
		velY: make([]float32, 0, count),
		accX: make([]float32, count),
		accY: make([]float32, count),
		mass: make([]float32, 0, count),
	}
}

func createParticles() particles {
	count := *countPtr
	speed := float32(*speedPtr)
	ringCount := *ringPtr

	ps := allocParticles(count)

	deltaDeg := 360.0 / float64(count/ringCount) * math.Pi / 180.0

	for i := range count {
		angle := (float64(i/ringCount) + float64(i%ringCount)/float64(ringCount)) * deltaDeg
		rad := SPAWN_RADIUS - SPAWN_RADIUS*float32(i%ringCount)/float32(ringCount)
		dirX := rad * float32(math.Cos(angle))
		dirY := rad * float32(math.Sin(angle))

		posX := dirX + float32(rl.GetScreenWidth())/2.0
		ps.posX = append(ps.posX, posX)
		posY := dirY + float32(rl.GetScreenHeight())/2.0
		ps.posY = append(ps.posY, posY)

		velX := dirY * speed
		ps.velX = append(ps.velX, velX)
		velY := -dirX * speed
		ps.velY = append(ps.velY, velY)

		mass := float32(5)
		ps.mass = append(ps.mass, mass)
		angle += deltaDeg
	}

	return ps
}

func updateSystemSimd(ps particles, lanes int, dt float32, p *pool) {
	halfKickDrift(ps, lanes, dt)
	calcAcceleration(ps, p)
	halfKick(ps, lanes, dt)
}

func halfKickDrift(ps particles, lanes int, dt float32) {
	half_dt := simd.BroadcastFloat32s(dt * 0.5)
	full_dt := simd.BroadcastFloat32s(dt)

	i := 0
	for i <= len(ps.posX)-lanes {
		// v(t + dt/2) = v(t) + a(t) * dt/2
		velX := simd.LoadFloat32s(ps.velX[i:])
		accX := simd.LoadFloat32s(ps.accX[i:])
		velX = accX.MulAdd(half_dt, velX)

		velY := simd.LoadFloat32s(ps.velY[i:])
		accY := simd.LoadFloat32s(ps.accY[i:])
		velY = accY.MulAdd(half_dt, velY)

		// x(t + dt) = x(t) + v(t + dt/2) * dt
		posX := simd.LoadFloat32s(ps.posX[i:])
		posX = velX.MulAdd(full_dt, posX)

		posY := simd.LoadFloat32s(ps.posY[i:])
		posY = velY.MulAdd(full_dt, posY)

		velX.Store(ps.velX[i:])
		velY.Store(ps.velY[i:])
		posX.Store(ps.posX[i:])
		posY.Store(ps.posY[i:])

		i += lanes
	}

	for ; i < len(ps.posX); i++ {
		ps.velX[i] += ps.accX[i] * dt * 0.5
		ps.velY[i] += ps.accY[i] * dt * 0.5

		ps.posX[i] += ps.velX[i] * dt
		ps.posY[i] += ps.velY[i] * dt
	}
}

func calcAcceleration(ps particles, p *pool) {
	clear(ps.accX)
	clear(ps.accY)
	p.run()
}

func calcAccelerationRange(ps particles, lanes, start, end int) {
	GConst := simd.BroadcastFloat32s(G)
	softeningSquared := simd.BroadcastFloat32s(GRAVITY_SOFTENINING * GRAVITY_SOFTENINING)
	zeroVec := simd.BroadcastFloat32s(0.0)
	oneVec := simd.BroadcastFloat32s(1.0)

	n := len(ps.posX)

	for i := start; i < end; i++ {
		p1PosXScalar := ps.posX[i]
		p1PosYScalar := ps.posY[i]

		p1AccXScalar := float32(0.0)
		p1AccYScalar := float32(0.0)

		p1PosX := simd.BroadcastFloat32s(ps.posX[i])
		p1PosY := simd.BroadcastFloat32s(ps.posY[i])

		p1AccX := zeroVec
		p1AccY := zeroVec

		j := 0
		for ; j < n-lanes; j += lanes {
			p2PosX := simd.LoadFloat32s(ps.posX[j:])
			p2PosY := simd.LoadFloat32s(ps.posY[j:])
			p2Mass := simd.LoadFloat32s(ps.mass[j:])

			dx := p2PosX.Sub(p1PosX)
			dy := p2PosY.Sub(p1PosY)
			distSqrd := dx.MulAdd(dx, dy.MulAdd(dy, softeningSquared))

			invDist := oneVec.Div(distSqrd.Sqrt())
			invDistCubed := invDist.Mul(invDist.Mul(invDist))

			commonFactor := invDistCubed.Mul(GConst)

			p1AccX = commonFactor.MulAdd(p2Mass.Mul(dx), p1AccX)
			p1AccY = commonFactor.MulAdd(p2Mass.Mul(dy), p1AccY)
		}

		for ; j < n; j++ {
			p2PosXScalar := ps.posX[j]
			p2PosYScalar := ps.posY[j]

			dx := p2PosXScalar - p1PosXScalar
			dy := p2PosYScalar - p1PosYScalar
			distSqrd := dx*dx + dy*dy + (GRAVITY_SOFTENINING * GRAVITY_SOFTENINING)
			dist := math.Sqrt(float64(distSqrd))

			invDist := 1.0 / dist
			invDistCubed := float32(invDist * invDist * invDist)

			p1AccXScalar += G * ps.mass[j] * dx * invDistCubed
			p1AccYScalar += G * ps.mass[j] * dy * invDistCubed
		}

		ps.accX[i] = sumFloat32s(p1AccX) + p1AccXScalar
		ps.accY[i] = sumFloat32s(p1AccY) + p1AccYScalar
	}
}

func sumFloat32s(v simd.Float32s) float32 {
	var buf [64]float32
	n := v.Len()
	v.Store(buf[:n])

	var sum float32
	for i := 0; i < n; i++ {
		sum += buf[i]
	}
	return sum
}

func halfKick(ps particles, lanes int, dt float32) {
	half_dt := simd.BroadcastFloat32s(dt * 0.5)

	i := 0
	for i <= len(ps.posX)-lanes {
		// v(t + dt/2) = v(t) + a(t) * dt/2
		velX := simd.LoadFloat32s(ps.velX[i:])
		accX := simd.LoadFloat32s(ps.accX[i:])
		velX = accX.MulAdd(half_dt, velX)

		velY := simd.LoadFloat32s(ps.velY[i:])
		accY := simd.LoadFloat32s(ps.accY[i:])
		velY = accY.MulAdd(half_dt, velY)

		velX.Store(ps.velX[i:])
		velY.Store(ps.velY[i:])

		i += lanes
	}

	for ; i < len(ps.posX); i++ {
		ps.velX[i] += ps.accX[i] * dt * 0.5
		ps.velY[i] += ps.accY[i] * dt * 0.5
	}
}

func drawParticles(ps particles) {
	for i := range ps.posX {
		rl.DrawCircle(int32(ps.posX[i]), int32(ps.posY[i]), ps.mass[i]*RADIUS_SCALE, rl.RayWhite)
	}
}

func main() {
	flag.Parse()
	rl.InitWindow(int32(rl.GetScreenWidth()), int32(rl.GetScreenHeight()), "Gravity Simulation - SoA")
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	particles := createParticles()

	var accumulator float32 = 0.0

	var vec simd.Float32s
	lanes := vec.Len()

	numWorkers := runtime.GOMAXPROCS(0)
	n := len(particles.posX)
	pool := newPool(n, numWorkers)
	pool.spawn(particles, lanes)
	defer pool.close()

	timer := time.NewTimer(2 * time.Second)
	var elapsed time.Duration

	for !rl.WindowShouldClose() {
		dt := rl.GetFrameTime()
		accumulator += dt

		now := time.Now()
		for accumulator >= PHYSICS_DT {
			updateSystemSimd(particles, lanes, PHYSICS_DT, pool)

			accumulator -= PHYSICS_DT
		}

		select {
		case <-timer.C:
			elapsed = time.Since(now)
			timer.Reset(2 * time.Second)
		default:
		}

		rl.BeginDrawing()
		rl.ClearBackground(rl.Black)

		// drawParticles(particles)

		// rl.DrawCircleLines(int32(rl.GetScreenWidth()/2.0), int32(rl.GetScreenHeight()/2.0), SPAWN_RADIUS, rl.RayWhite)

		rl.DrawText(fmt.Sprintf("simd lanes: %d", lanes), 10, 30, 20, rl.RayWhite)
		rl.DrawText(fmt.Sprintf("update time: %.1f ms", float64(elapsed.Milliseconds())), 10, 50, 20, rl.RayWhite)
		rl.DrawText(fmt.Sprintf("number of workers: %d", numWorkers), 10, 70, 20, rl.RayWhite)
		rl.DrawFPS(10, 10)

		rl.EndDrawing()
	}
}
