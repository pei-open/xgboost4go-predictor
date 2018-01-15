package math

import (
	"unsafe"
	"math"
)

const (
	MAX_FLOAT32 = 3.40282346638528859811704183484516925440e+38                              // 2**127 * (2**24 - 1) / 2**23
	E           = float32(2.71828182845904523536028747135266249775724709369995957496696763) // http://oeis.org/A001113
)

const (
	uvnan    = 0x7FE00000
	uvinf    = 0x7F800000
	uvneginf = 0xFF800000
	mask     = 0xFF
	shift    = 32 - 8 - 1
	bias     = 127
)

var NAN = Float32frombits(uvnan)

func MaxFloat32(x, y float32) float32 {
	switch {
	case IsInf(x, 1) || IsInf(y, 1):
		return Inf(1)
	case IsNaN(x) || IsNaN(y):
		return NaN()
	case x == 0 && x == y:
		if Signbit(x) {
			return y
		}
		return x
	}
	if x > y {
		return x
	}
	return y
}

func IsInf(f float32, sign int) bool {
	// Test for infinity by comparing against maximum float.
	// To avoid the floating-point hardware, could use:
	// x := Float32bits(f)
	// return sign >= 0 && x == uvinf || sign <= 0 && x == uvneginf
	return sign >= 0 && f > MAX_FLOAT32 || sign <= 0 && f < -MAX_FLOAT32
}

func Inf(sign int) float32 {
	var v uint32
	if sign >= 0 {
		v = uvinf
	} else {
		v = uvneginf
	}
	return Float32frombits(v)
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() float32 { return NAN }

// IsNaN reports whether f is an IEEE 754 ``not-a-number'' value.
func IsNaN(f float32) (is bool) {
	// IEEE 754 says that only NaNs satisfy f != f.
	// To avoid the floating-point hardware, could use:
	// x := Float32bits(f)
	// return uint32(x>>shift)&mask == mask && x != uvinf && x != uvneginf
	return f != f
}

// Float32bits returns the IEEE 754 binary representation of f.
func Float32bits(f float32) uint32 { return *(*uint32)(unsafe.Pointer(&f)) }

// Float32frombits returns the floating point number corresponding
// to the IEEE 754 binary representation b.
func Float32frombits(b uint32) float32 { return *(*float32)(unsafe.Pointer(&b)) }

// Float64bits returns the IEEE 754 binary representation of f.
func Float64bits(f float64) uint64 { return *(*uint64)(unsafe.Pointer(&f)) }

// Float64frombits returns the floating point number corresponding
// the IEEE 754 binary representation b.
func Float64frombits(b uint64) float64 { return *(*float64)(unsafe.Pointer(&b)) }

// Signbit returns true if x is negative or negative zero.
func Signbit(x float32) bool {
	return Float32bits(x)&(1<<31) != 0
}

func ExpFloat32(x float32) float32 {
	const (
		Ln2Hi = float32(6.9313812256e-01)
		Ln2Lo = float32(9.0580006145e-06)
		Log2e = float32(1.4426950216e+00)

		Overflow  = 7.09782712893383973096e+02
		Underflow = -7.45133219101941108420e+02
		NearZero  = 1.0 / (1 << 28) // 2**-28

		LogMax = 0x42b2d4fc // The bitmask of log(FLT_MAX), rounded down.  This value is the largest input that can be passed to exp() without producing overflow.
		LogMin = 0x42aeac50 // The bitmask of |log(REAL_FLT_MIN)|, rounding down

	)
	// hx := Float32bits(x) & uint32(0x7fffffff)

	// special cases
	switch {
	case IsNaN(x) || IsInf(x, 1):
		return x
	case IsInf(x, -1):
		return 0
	case x > Overflow:
		return Inf(1)
	case x < Underflow:
		return 0
		// case hx > LogMax:
		// 	return Inf(1)
		// case x < 0 && hx > LogMin:
		return 0
	case -NearZero < x && x < NearZero:
		return 1 + x
	}

	// reduce; computed as r = hi - lo for extra precision.
	var k int
	switch {
	case x < 0:
		k = int(Log2e*x - 0.5)
	case x > 0:
		k = int(Log2e*x + 0.5)
	}
	hi := x - float32(k)*Ln2Hi
	lo := float32(k) * Ln2Lo

	// compute
	return expmulti(hi, lo, k)
}

// Exp2 returns 2**x, the base-2 exponential of x.
//
// Special cases are the same as Exp.
func Exp2Float32(x float32) float32 {
	const (
		Ln2Hi = 6.9313812256e-01
		Ln2Lo = 9.0580006145e-06

		Overflow  = 1.0239999999999999e+03
		Underflow = -1.0740e+03
	)

	// special cases
	switch {
	case IsNaN(x) || IsInf(x, 1):
		return x
	case IsInf(x, -1):
		return 0
	case x > Overflow:
		return Inf(1)
	case x < Underflow:
		return 0
	}

	// argument reduction; x = r×lg(e) + k with |r| ≤ ln(2)/2.
	// computed as r = hi - lo for extra precision.
	var k int
	switch {
	case x > 0:
		k = int(x + 0.5)
	case x < 0:
		k = int(x - 0.5)
	}
	t := x - float32(k)
	hi := t * Ln2Hi
	lo := -t * Ln2Lo

	// compute
	return expmulti(hi, lo, k)
}

// exp1 returns e**r × 2**k where r = hi - lo and |r| ≤ ln(2)/2.
func expmulti(hi, lo float32, k int) float32 {
	const (
		P1 = float32(1.6666667163e-01)  /* 0x3e2aaaab */
		P2 = float32(-2.7777778450e-03) /* 0xbb360b61 */
		P3 = float32(6.6137559770e-05)  /* 0x388ab355 */
		P4 = float32(-1.6533901999e-06) /* 0xb5ddea0e */
		P5 = float32(4.1381369442e-08)  /* 0x3331bb4c */
	)

	r := hi - lo
	t := r * r
	c := r - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
	y := 1 - ((lo - (r*c)/(2-c)) - hi)
	// TODO(rsc): make sure Ldexp can handle boundary k
	return Ldexp(y, k)
}

func Ldexp(frac float32, exp int) float32 {
	// special cases
	switch {
	case frac == 0:
		return frac // correctly return -0
	case IsInf(frac, 0) || IsNaN(frac):
		return frac
	}
	frac, e := normalize(frac)
	exp += e
	x := Float32bits(frac)
	exp += int(x>>shift)&mask - bias
	if exp < -149 {
		return Copysign(0, frac) // underflow
	}
	if exp > 127 { // overflow
		if frac < 0 {
			return Inf(-1)
		}
		return Inf(1)
	}
	var m float32 = 1
	if exp < -(127 - 1) { // denormal
		exp += shift
		m = 1.0 / (1 << 23) // 1/(2**-23)
	}
	x &^= mask << shift
	x |= uint32(exp+bias) << shift
	return m * Float32frombits(x)
}

// normalize returns a normal number y and exponent exp
// satisfying x == y × 2**exp. It assumes x is finite and non-zero.
func normalize(x float32) (y float32, exp int) {
	const SmallestNormal = 1.1754943508222875079687365e-38 // 2**-(127 - 1)
	if Abs(x) < SmallestNormal {
		return x * (1 << shift), -shift
	}
	return x, 0
}

// Abs returns the absolute value of x.
//
// Special cases are:
//	Abs(±Inf) = +Inf
//	Abs(NaN) = NaN
func Abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	if x == 0 {
		return 0 // return correctly abs(-0)
	}
	return x

	// asUint := Float32bits(x) & uint32(0x7FFFFFFF)
	// return Float32frombits(asUint)
}

func Copysign(x, y float32) float32 {
	const sign = 1 << 31
	return math.Float32frombits(math.Float32bits(x)&^sign | math.Float32bits(y)&sign)
}
