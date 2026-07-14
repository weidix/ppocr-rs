#!/bin/sh
set -eu

root=${1:-models}

if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI is required" >&2
    exit 1
fi

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download() {
    name=$1
    repository=$2
    revision=$3
    expected=$4
    directory="$root/$name"

    hf download "$repository" \
        model.safetensors config.json inference.yml preprocessor_config.json \
        --revision "$revision" \
        --local-dir "$directory"

    actual=$(sha256_file "$directory/model.safetensors")
    if [ "$actual" != "$expected" ]; then
        echo "$name checksum mismatch: expected $expected, found $actual" >&2
        exit 1
    fi
    echo "$name $actual"
}

download medium-det PaddlePaddle/PP-OCRv6_medium_det_safetensors \
    4236c2b61741a259c091fd879dcc4edc339e916c \
    bd393266c02e1a680b1b34c301d5d0d81e6290440b7f8ab0f5d5032276b17eb1
download medium-rec PaddlePaddle/PP-OCRv6_medium_rec_safetensors \
    024cad6a831de75c2c3c26e711ba8c4a82ccd24b \
    5f43c16f2a684b1d2284662178bdb604febd3d6bfdb5ca73828d08d0f7c0c3e9
download small-det PaddlePaddle/PP-OCRv6_small_det_safetensors \
    eae2ee920a39fb3087637d3dbb58df1896ec1f24 \
    89a96a8adc4e9cd0c994098edc76022e496d35844392562b4694c8fbc583f2da
download small-rec PaddlePaddle/PP-OCRv6_small_rec_safetensors \
    fe049fb103f57443fe8840c54ed06b702f3c1de5 \
    f65a332afe5aa663f0b9d5706f4ae8457b5b4058a842d5c1eb22df505c27d642
download tiny-det PaddlePaddle/PP-OCRv6_tiny_det_safetensors \
    07595f982703daf0d4e120a12a01da8073542f3a \
    cae3c88d2a9902fd0293e6b17990428f54bfa7ec98f800a4368e95423a754d16
download tiny-rec PaddlePaddle/PP-OCRv6_tiny_rec_safetensors \
    6f2d2d51b4b4226d7a2329a02f416f4994106f3a \
    cc3892aba0fbd89afbf6a76d8b7817bb58802668be7a1384ca761ce65612f3f7
