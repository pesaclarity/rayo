// Compiles rayo and the vendored ggml (third_party/ggml) from source into one
// shared library, for whatever target Dart or Flutter is building: Android
// (arm64-v8a, armeabi-v7a, x86_64), iOS, Linux and macOS. Nothing is
// downloaded: every source file is in this package (see third_party/ggml/PIN).

import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

const _ggml = 'third_party/ggml';

/// ggml's C sources (ggml-base and ggml-cpu).
List<String> _cSources(String arch) => [
  '$_ggml/src/ggml.c',
  '$_ggml/src/ggml-alloc.c',
  '$_ggml/src/ggml-quants.c',
  '$_ggml/src/ggml-cpu/ggml-cpu.c',
  '$_ggml/src/ggml-cpu/quants.c',
  if (arch.isNotEmpty) '$_ggml/src/ggml-cpu/arch/$arch/quants.c',
];

/// ggml's C++ sources.
List<String> _cppSources(String arch, {required bool sgemm}) => [
  '$_ggml/src/ggml.cpp',
  '$_ggml/src/ggml-backend.cpp',
  '$_ggml/src/ggml-backend-meta.cpp',
  '$_ggml/src/ggml-opt.cpp',
  '$_ggml/src/ggml-threading.cpp',
  '$_ggml/src/gguf.cpp',
  '$_ggml/src/ggml-cpu/binary-ops.cpp',
  '$_ggml/src/ggml-cpu/ggml-cpu.cpp',
  '$_ggml/src/ggml-cpu/ops.cpp',
  '$_ggml/src/ggml-cpu/repack.cpp',
  '$_ggml/src/ggml-cpu/traits.cpp',
  '$_ggml/src/ggml-cpu/unary-ops.cpp',
  '$_ggml/src/ggml-cpu/vec.cpp',
  '$_ggml/src/ggml-cpu/tiled/tiled.cpp',
  '$_ggml/src/ggml-cpu/tiled/tiled-kernel.cpp',
  if (sgemm) '$_ggml/src/ggml-cpu/llamafile/sgemm.cpp',
  if (arch.isNotEmpty) '$_ggml/src/ggml-cpu/arch/$arch/repack.cpp',
];

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final code = input.config.code;
    final os = code.targetOS;
    final cpu = code.targetArchitecture;
    final arch = switch (cpu) {
      Architecture.arm64 || Architecture.arm => 'arm',
      Architecture.x64 || Architecture.ia32 => 'x86',
      _ => '',
    };
    final apple = os == OS.iOS || os == OS.macOS;
    // The llamafile sgemm needs 64-bit ARM's float16 vectors; 32-bit ARM
    // phones use ggml's own matrix products.
    final sgemm = cpu != Architecture.arm;

    final defines = <String, String?>{
      'GGML_SCHED_MAX_COPIES': '4',
      'GGML_USE_CPU_REPACK': null,
      if (sgemm) 'GGML_USE_LLAMAFILE': null,
      if (apple)
        '_DARWIN_C_SOURCE': null
      else ...{
        '_GNU_SOURCE': null,
        '_XOPEN_SOURCE': '600',
      },
      // The Linux build runs the tests on the computer; it assumes AVX2
      // (x86-64-v3), as every x86 computer of the last ten years has.
      if (os == OS.linux && cpu == Architecture.x64) ...{
        'GGML_AVX': null,
        'GGML_AVX2': null,
        'GGML_FMA': null,
        'GGML_F16C': null,
        'GGML_BMI2': null,
        'GGML_SSE42': null,
      },
    };
    final flags = <String>[
      '-ffunction-sections',
      '-fdata-sections',
      '-fvisibility=hidden',
      '-Wno-unused-command-line-argument',
      if (os == OS.linux && cpu == Architecture.x64) ...[
        '-msse4.2',
        '-mavx',
        '-mavx2',
        '-mfma',
        '-mf16c',
        '-mbmi2',
      ],
      if (os == OS.android && cpu == Architecture.arm) ...['-mfpu=neon'],
      // Reproducible: no build paths in the binary.
      '-ffile-prefix-map=${input.packageRoot.toFilePath()}=.',
    ];
    final includes = [
      '$_ggml/include',
      '$_ggml/src',
      '$_ggml/src/ggml-cpu',
      'src',
    ];

    // native_toolchain_c compiles one language per builder, so ggml's C and
    // C++ halves become two static libraries, linked into librayo.
    await CBuilder.library(
      name: 'rayo_ggml_c',
      sources: _cSources(arch),
      includes: includes,
      defines: defines,
      flags: flags,
      std: 'gnu11',
      linkModePreference: LinkModePreference.static,
    ).run(input: input, output: output);
    await CBuilder.library(
      name: 'rayo_ggml_cpp',
      sources: _cppSources(arch, sgemm: sgemm),
      includes: includes,
      defines: defines,
      flags: [...flags, '-fvisibility-inlines-hidden'],
      std: 'gnu++17',
      language: Language.cpp,
      linkModePreference: LinkModePreference.static,
    ).run(input: input, output: output);

    // Export the rayo_ functions and nothing else (ggml and the C++ runtime
    // stay inside), which also lets the linker drop what rayo does not use.
    String? versionScript;
    if (os == OS.linux || os == OS.android) {
      final file = File.fromUri(input.outputDirectory.resolve('rayo.map'));
      await file.writeAsString(
        '{\n  global:\n    rayo_*;\n  local:\n    *;\n};\n',
      );
      versionScript = file.path;
    }

    final ggml = ['rayo_ggml_cpp', 'rayo_ggml_c'];
    await CBuilder.library(
      name: 'rayo',
      assetName: 'src/ffi.dart',
      sources: ['src/rayo.cpp'],
      includes: includes,
      defines: defines,
      flags: [
        ...flags,
        '-fvisibility-inlines-hidden',
        if (apple) '-Wl,-dead_strip' else '-Wl,--gc-sections',
        if (os == OS.linux || os == OS.android) ...[
          '-Wl,--no-undefined',
          '-Wl,-Bsymbolic',
          '-Wl,--version-script=$versionScript',
          '-Wl,-z,relro,-z,now,-z,noexecstack',
          '-Wl,--build-id=none',
        ],
        if (apple) ...['-install_name', '@rpath/librayo.dylib'],
      ],
      std: 'gnu++17',
      language: Language.cpp,
      cppLinkStdLib: os == OS.android ? 'c++_static' : null,
      // Twice: the two halves call each other.
      libraries: [
        ...ggml,
        ...ggml,
        if (os == OS.linux || os == OS.android) 'm',
        if (os == OS.linux) 'pthread',
      ],
      linkModePreference: LinkModePreference.dynamic,
    ).run(input: input, output: output);
  });
}
