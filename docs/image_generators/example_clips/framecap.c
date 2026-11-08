// LD_PRELOAD frame capture for SDL2 + OpenGL programs.
//
// Before each SDL_GL_SwapWindow, the finished frame in the default framebuffer
// is read back and written to an ffmpeg pipe at a fixed rate of wall-clock time
// (slow frames are repeated, extra frames dropped), so the clip plays in real
// time whatever the program's frame rate. Works with SDL_VIDEODRIVER=offscreen.
//
//   FRAMECAP_OUT      output .mp4 (required)
//   FRAMECAP_START    seconds after the first frame before recording (default 3)
//   FRAMECAP_SECONDS  clip length in seconds (default 6)
//   FRAMECAP_FPS      output frame rate (default 30)
//   FRAMECAP_WIDTH    output width; height keeps the aspect ratio (default 960, 0: native)
//   FRAMECAP_CRF      x264 quality (default 23)
//   FRAMECAP_EVERY_FRAME  1: keep every frame instead of sampling wall-clock time,
//                     for programs that advance their simulation per frame
//   FRAMECAP_EXIT     1: end the process once the clip is written
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct SDL_Window SDL_Window;
typedef unsigned int GLenum;
typedef int GLint;
typedef int GLsizei;

#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_READ_FRAMEBUFFER_BINDING 0x8CAA
#define GL_PIXEL_PACK_BUFFER 0x88EB
#define GL_PIXEL_PACK_BUFFER_BINDING 0x88ED
#define GL_READ_BUFFER 0x0C02
#define GL_PACK_ALIGNMENT 0x0D05
#define GL_BACK 0x0405
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401

static void (*real_swap)(SDL_Window*);
static void* (*get_proc)(const char*);
static void (*get_drawable_size)(SDL_Window*, int*, int*);
static void (*gl_get_integerv)(GLenum, GLint*);
static void (*gl_bind_framebuffer)(GLenum, unsigned);
static void (*gl_bind_buffer)(GLenum, unsigned);
static void (*gl_read_buffer)(GLenum);
static void (*gl_pixel_storei)(GLenum, GLint);
static void (*gl_read_pixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);

static FILE* pipe_out;
static unsigned char* pixels;
static int width, height, finished;
static long frames_written;
static double t0 = -1;

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double env_number(const char* name, double fallback) {
  const char* value = getenv(name);
  return value && *value ? atof(value) : fallback;
}

static int load_gl(void) {
  if (gl_read_pixels) return 1;
  get_proc = (void* (*)(const char*))dlsym(RTLD_DEFAULT, "SDL_GL_GetProcAddress");
  get_drawable_size = (void (*)(SDL_Window*, int*, int*))dlsym(RTLD_DEFAULT, "SDL_GL_GetDrawableSize");
  if (!get_proc || !get_drawable_size) return 0;
  gl_get_integerv = (void (*)(GLenum, GLint*))get_proc("glGetIntegerv");
  gl_bind_framebuffer = (void (*)(GLenum, unsigned))get_proc("glBindFramebuffer");
  gl_bind_buffer = (void (*)(GLenum, unsigned))get_proc("glBindBuffer");
  gl_read_buffer = (void (*)(GLenum))get_proc("glReadBuffer");
  gl_pixel_storei = (void (*)(GLenum, GLint))get_proc("glPixelStorei");
  gl_read_pixels = (void (*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*))get_proc("glReadPixels");
  return gl_get_integerv && gl_bind_framebuffer && gl_bind_buffer && gl_read_buffer && gl_pixel_storei &&
         gl_read_pixels;
}

static void read_frame(void) {
  GLint read_fbo, read_buffer, pack_buffer, pack_alignment;
  gl_get_integerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
  gl_get_integerv(GL_READ_BUFFER, &read_buffer);
  gl_get_integerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack_buffer);
  gl_get_integerv(GL_PACK_ALIGNMENT, &pack_alignment);
  gl_bind_framebuffer(GL_READ_FRAMEBUFFER, 0);
  gl_bind_buffer(GL_PIXEL_PACK_BUFFER, 0);
  gl_read_buffer(GL_BACK);
  gl_pixel_storei(GL_PACK_ALIGNMENT, 1);
  gl_read_pixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
  gl_pixel_storei(GL_PACK_ALIGNMENT, pack_alignment);
  gl_read_buffer((GLenum)read_buffer);
  gl_bind_buffer(GL_PIXEL_PACK_BUFFER, (unsigned)pack_buffer);
  gl_bind_framebuffer(GL_READ_FRAMEBUFFER, (unsigned)read_fbo);
}

static void open_pipe(void) {
  char command[2048], filter[128];
  const int out_width = (int)env_number("FRAMECAP_WIDTH", 960);
  if (out_width > 0) {
    snprintf(filter, sizeof(filter), "vflip,scale=%d:-2:flags=lanczos", out_width);
  } else {
    snprintf(filter, sizeof(filter), "vflip");
  }
  snprintf(command, sizeof(command),
           "ffmpeg -nostdin -loglevel error -y -f rawvideo -pix_fmt rgba -s %dx%d -r %g -i - "
           "-vf '%s' -an -c:v libx264 -preset slow -crf %d "
           "-pix_fmt yuv420p -profile:v high -movflags +faststart '%s'",
           width, height, env_number("FRAMECAP_FPS", 30), filter, (int)env_number("FRAMECAP_CRF", 23),
           getenv("FRAMECAP_OUT"));
  pipe_out = popen(command, "w");
  pixels = (unsigned char*)malloc((size_t)width * height * 4);
}

static void capture(SDL_Window* window) {
  if (finished || !getenv("FRAMECAP_OUT") || !load_gl()) return;
  const double t = now();
  if (t0 < 0) t0 = t;
  const double start = env_number("FRAMECAP_START", 3), seconds = env_number("FRAMECAP_SECONDS", 6);
  const double fps = env_number("FRAMECAP_FPS", 30), elapsed = t - t0 - start;
  if (elapsed < 0) return;
  if (!pipe_out) {
    get_drawable_size(window, &width, &height);
    width &= ~1;
    height &= ~1;
    open_pipe();
  }
  int w, h;
  get_drawable_size(window, &w, &h);
  if ((w & ~1) != width || (h & ~1) != height) return;
  const long total = (long)(seconds * fps);
  const long target = env_number("FRAMECAP_EVERY_FRAME", 0) > 0
                          ? (frames_written < total ? frames_written + 1 : total)
                          : (elapsed >= seconds ? total : (long)(elapsed * fps) + 1);
  if (frames_written < target) {
    read_frame();
    for (; frames_written < target; ++frames_written) fwrite(pixels, 1, (size_t)width * height * 4, pipe_out);
  }
  if (frames_written >= total) {
    finished = 1;
    pclose(pipe_out);
    fprintf(stderr, "framecap: wrote %ld frames (%dx%d) to %s\n", frames_written, width, height,
            getenv("FRAMECAP_OUT"));
    if (env_number("FRAMECAP_EXIT", 0) > 0) _exit(0);
  }
}

void SDL_GL_SwapWindow(SDL_Window* window) {
  if (!real_swap) real_swap = (void (*)(SDL_Window*))dlsym(RTLD_NEXT, "SDL_GL_SwapWindow");
  capture(window);
  real_swap(window);
}
