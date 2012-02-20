package savepng
//#cgo pkg-config: sdl SDL_image libpng
// #include "SDL.h"
// #include "IMG_savepng.h"
import "C"

import (
	"github.com/banthar/Go-SDL/sdl"
	"unsafe"
)

func SavePNG(file string, s *sdl.Surface, compression int) int {
	cfile := C.CString(file)
	defer C.free(unsafe.Pointer(cfile))
	res := C.IMG_SavePNG(cfile, (*C.SDL_Surface)(unsafe.Pointer(s)), C.int(compression))
	return int(res)
}
