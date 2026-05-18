{
  pkgs ? import <nixpkgs> { },
}:

let
  runtimeLibs = with pkgs; [
    # C/C++ runtime
    stdenv.cc.cc.lib
    zlib

    # OpenCV / ultralytics common deps
    glib
    libGL
    libglvnd
    mesa

    # X11 / XCB
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libxcb
    xorg.libXau
    xorg.libXdmcp
    xorg.libSM
    xorg.libICE
    xorg.libXi
    xorg.libXrandr
    xorg.libXfixes
    xorg.libXcursor
    xorg.libXinerama

    # Often needed by wheels
    freetype
    fontconfig
    expat
    libpng
    libjpeg
    libtiff
    libwebp
    openjpeg

    # Misc runtime libs
    libffi
    openssl
    curl
    sqlite
    bzip2
    xz
    lz4
    zstd
  ];
in

pkgs.mkShell {
  packages =
    with pkgs;
    [
      python312
      python312Packages.pip
      python312Packages.virtualenv
      python312Packages.tkinter
    ]
    ++ runtimeLibs;

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"

    source ~/venvs/ml/bin/activate
    echo "Using Python: $(which python)"
  '';
}
