project('sil', 'c',
  version: '0.1.0',
  meson_version: '>=0.55.0',
  default_options: ['c_std=c11', 'warning_level=3']
)

public_headers = files(
  'include/libsil.h',
)

sources = files(
  'src/sil.c',
)

include_dirs = include_directories('include')

sil_library = static_library('sil',
  sources: sources,
  include_directories: include_dirs,
  install: true
)

sil_cli = executable('sil',
  sources: 'src/cli.c',
  link_with: sil_library,
  include_directories: include_dirs,
  install: true
)

install_headers(public_headers, subdir: 'sil')
