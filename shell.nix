{ pkgs ? import <nixpkgs> {} }:

let
  python-with-packages = pkgs.python311.withPackages (ps: with ps; [
    beautifulsoup4
    requests
    numpy
    faiss  # Consider faiss-rocm if you want GPU acceleration
  ]);
in
pkgs.mkShell {
  buildInputs = [
    # System dependencies
    python-with-packages
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.gcc
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libffi
    pkgs.xz
    pkgs.bzip2

    # ROCm stack
#    pkgs.rocmPackages.rocm-core
#    pkgs.rocmPackages.rocminfo

#    pkgs.rocmPackages.rocm-runtime
#    pkgs.rocmPackages.rocm-device-libs
#    pkgs.rocmPackages.clr
#    pkgs.rocmPackages.hipblas
#    pkgs.rocmPackages.rocblas
#    pkgs.rocmPackages.rocm-smi
  ];

  shellHook = ''
    VENV_DIR=".venv"
    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment in $VENV_DIR"
      python -m venv $VENV_DIR
      source $VENV_DIR/bin/activate
      
      # Install PyTorch with ROCm6 support first
      pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0
      
      # Then other packages
      pip install transformers sentence-transformers
    else
      source $VENV_DIR/bin/activate
    fi
    
    # Critical ROCm environment variables for RX 5700 XT
    # export HSA_OVERRIDE_GFX_VERSION=10.3.0
    # export HIP_VISIBLE_DEVICES=0
    # export ROCR_VISIBLE_DEVICES=0
    # export HCC_AMDGPU_TARGET=gfx1030
    
    # Library paths
    # export LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib:${pkgs.rocmPackages.rocblas}/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libffi}/lib:${pkgs.xz}/lib:${pkgs.bzip2}/lib:${pkgs.rocmPackages.rocm-runtime}/lib:$LD_LIBRARY_PATH
  '';
}
