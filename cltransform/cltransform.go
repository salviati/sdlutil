package cltransform
import (
	"errors"
	"github.com/salviati/go-opencl/cl"
	"github.com/banthar/Go-SDL/sdl"
	"sync"
)

const kernelSource string = `
constant sampler_t linear  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
constant sampler_t nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// recfactorx = 1/factorx ---taking a division outside of the "loop".
__kernel void image_recscale(__read_only image2d_t src, __write_only image2d_t dst, float recfactorx, float recfactory) {
	int2 p = {get_global_id(0), get_global_id(1)};
	float2 q = {convert_float(p.x)*recfactorx, convert_float(p.y)*recfactory};

	uint4 pixel = read_imageui(src, linear, q);
	write_imageui(dst, p, pixel);
}

__kernel void image_rotate(__read_only  image2d_t src, __write_only image2d_t dst, float s, float c) {
	int2 p = {get_global_id(0), get_global_id(1)};
	int2 o = {get_image_width(src)/2, get_image_height(src)/2};

	float2 po = convert_float2(p-o);

	float2 qo = {c*po.x - s*po.y, s*po.x + c*po.y};
	float2 q = (qo) + convert_float2(o);

	uint4 pixel = read_imageui(src, linear, q);
	write_imageui(dst, p, pixel);
}

__kernel void image_flip_h(__read_only  image2d_t src, __write_only image2d_t dst) {
	int2 p = {get_global_id(0), get_global_id(1)};
	int2 q = {get_image_width(src) - p.x, p.y};
	
	uint4 pixel = read_imageui(src, nearest, q);
	write_imageui(dst, p, pixel);
}

__kernel void image_flip_v(__read_only  image2d_t src, __write_only image2d_t dst) {
	int2 p = {get_global_id(0), get_global_id(1)};
	int2 q = {p.x, get_image_height(src) - p.y};
	
	uint4 pixel = read_imageui(src, nearest, q);
	write_imageui(dst, p, pixel);
}

__kernel void image_flip_hv(__read_only  image2d_t src, __write_only image2d_t dst) {
	int2 p = {get_global_id(0), get_global_id(1)};
	int2 q = {get_image_width(src) - p.x, get_image_height(src) - p.y};
	
	uint4 pixel = read_imageui(src, nearest, q);
	write_imageui(dst, p, pixel);
}

// q = A.p
__kernel void image_affine(__read_only  image2d_t src, __write_only image2d_t dst, float2 Ax, float2 Ay) {
	int2 p = {get_global_id(0), get_global_id(1)};
	float2 pf = convert_float2(p);
	float2 q = {dot(Ax, pf), dot(Ay, pf)};

	uint4 pixel = read_imageui(src, linear, q);
	write_imageui(dst, p, pixel);
}

// q-q0 = A.(p-p0)
__kernel void image_affine2(__read_only  image2d_t src, __write_only image2d_t dst, float2 Ax, float2 Ay, float2 p0, float2 q0) {
	int2 p = {get_global_id(0), get_global_id(1)};
	float2 pf = convert_float2(p);
	pf -= p0;
	float2 q = {dot(Ax, pf), dot(Ay, pf)};
	q += q0;

	uint4 pixel = read_imageui(src, linear, q);
	write_imageui(dst, p, pixel);
}`

var kernelNames = []string{
	"image_recscale", "image_rotate",
	"image_flip_h", "image_flip_v", "image_flip_hv",
	"image_affine","image_affine2",
}

type Env struct {
	platform cl.Platform
	c        *cl.Context
	cq       *cl.CommandQueue
	p        *cl.Program
	kernels  map[string]*cl.Kernel
	lock     sync.Mutex
}


// FIXME(utkan): let user choose other platform/device.
func NewEnv() (*Env, error)  {
	e := new(Env)
	
	e.kernels = make(map[string]*cl.Kernel)

	platforms := cl.GetPlatforms()
	e.platform = platforms[0]

	if e.platform.Devices[0].Property(cl.DEVICE_IMAGE_SUPPORT).(bool) == false {
		return nil, errors.New("Your device doesn't support images through OpenCL")
	}
	
	err := e.initAndPrepCL()
	
	return e, err
}

// init OpenCL & load the program
func (e *Env) initAndPrepCL() error {
	var err error

	params := make(map[cl.ContextParameter]interface{})
	e.c, err = cl.NewContextOfDevices(params, e.platform.Devices[0:1])
	if err != nil {
		return err
	}

	e.cq, err = e.c.NewCommandQueue(e.platform.Devices[0], cl.QUEUE_PROFILING_ENABLE|cl.QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
	if err != nil {
		return err
	}

	e.p, err = e.c.NewProgramFromSource(kernelSource)
	if err != nil {
		return err
	}

	addKernel := func(name string) (err error) {
		e.kernels[name], err = e.p.NewKernelNamed(name)
		return
	}

	for _, kernelName := range kernelNames {
		if err := addKernel(kernelName); err != nil {
			return err
		}
	}

	return nil
}

// Will call kernelName(src, dst, va...) on the device side.
func (e *Env) call(dstW, dstH uint32, kernelName string, src, dst *cl.Image, va ...interface{}) ([]byte, error) {
	k, ok := e.kernels[kernelName]
	if !ok {
		return nil, errors.New("Unknown kernel " + kernelName)
	}

	err := k.SetArg(0, src)
	if err != nil {
		return nil, err
	}

	err = k.SetArg(1, dst)
	if err != nil {
		return nil, err
	}

	for i, v := range va {
		err = k.SetArg(uint(i+2), v)
		if err != nil {
			return nil, err
		}
	}

	empty := make([]cl.Size, 0)
	gsize := []cl.Size{cl.Size(dstW), cl.Size(dstH)}
	err = e.cq.EnqueueKernel(k, empty, gsize, empty)
	if err != nil {
		return nil, err
	}
	
	pixels, err := e.cq.EnqueueReadImage(dst, [3]cl.Size{0, 0, 0}, [3]cl.Size{cl.Size(dstW), cl.Size(dstH), 1}, 0, 0)
	if err != nil {
		return nil, err
	}

	return pixels, nil
}

type Image struct {
	clImage *cl.Image
	
	
	s *sdl.Surface
	
	// used by write-only images
	pixels []byte
	w,h uint32
}

// Creates a new image for writing purposes.
func (e *Env) NewWImage(w, h uint32) (*Image, error) {
	order := cl.RGB //FIXME
	img, err := e.c.NewImage2D(cl.MEM_WRITE_ONLY, order, cl.UNSIGNED_INT8,
		w, h, 0, nil)

	if err != nil {
		return nil, err
	}
	
	return &Image{clImage: img, w: w, h: h}, nil
}

// Creates a new, read-only image whose contents are
// s.Pixels. A copy of SDL surface pointer remains in the
// image struct, so you don't need to worry about the garbage collector.
// Just don't free it until you're done with this Image.
func (e *Env) NewImage(s *sdl.Surface) (*Image, error) {
	if s.Format.BytesPerPixel != 3 { //FIXME
		return nil, errors.New("Unsupported image format")
	}

	order := cl.RGB //FIXME

	flags := cl.MEM_COPY_HOST_PTR
	//flags = cl.MEM_ALLOC_HOST_PTR
	clImg, err := e.c.NewImage2D(cl.MEM_READ_ONLY|flags, order, cl.UNSIGNED_INT8,
			uint32(s.W), uint32(s.H), uint32(s.Pitch), s.Pixels)

	if err != nil {
		return nil, err
	}
	
	return &Image{s:s, clImage: clImg}, nil
}

func (img *Image) createSDLSurface(f *sdl.PixelFormat) (*sdl.Surface, error) {
	pitch := uint32(f.BytesPerPixel)*img.w
	
	s := sdl.CreateRGBSurfaceFrom(img.pixels,
				int(img.w), int(img.h), int(f.BitsPerPixel), int(pitch),
				f.Rmask, f.Gmask, f.Bmask, f.Amask,
			)
	
	if s == nil {
			return nil, errors.New(sdl.GetError())
		}
	
	s.Pixels = &img.pixels[0] // Avoid garbage collection
	
	return s, nil
}

func (e *Env) callWrap(dstW, dstH uint32, kernelName string, s *sdl.Surface, va ...interface{}) (*sdl.Surface, error) {
	src, err := e.NewImage(s)
	if err != nil { return nil, err}

	dst, err := e.NewWImage(dstW, dstH)

	dst.pixels, err = e.call(dstW, dstH, kernelName, src.clImage, dst.clImage, va...)
	if err != nil { return nil, err}

	news, err := dst.createSDLSurface(s.Format)
	if err != nil { return nil, err}
	
	return news, nil
}

func (e *Env) Scale(s *sdl.Surface, factorx, factory float32) (*sdl.Surface, error) {
	w := uint32(float32(s.W)*factorx)
	h := uint32(float32(s.H)*factory)
	
	return e.callWrap(w, h, "image_recscale", s, float32(1)/factorx, float32(1)/factory)
}

func (e *Env) Flip(s *sdl.Surface, horizontal, vertical bool) (*sdl.Surface, error) {
	w, h := uint32(s.W), uint32(s.H)

	if horizontal && vertical {
		return e.callWrap(w, h, "image_flip_hv", s)
	}
	if horizontal {
		return e.callWrap(w, h, "image_flip_h", s)
	}
	if vertical {
		return e.callWrap(w, h, "image_flip_v", s)
	}

	return s, nil
}
