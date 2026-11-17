// Compiles rayo's C sources (src/) into one shared library for whatever
// target Dart or Flutter is building: Android (arm64-v8a, armeabi-v7a,
// x86_64), iOS, macOS and Linux. Plain C89, no other library, nothing
// downloaded.

import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final code = input.config.code;
    final os = code.targetOS;
    final cpu = code.targetArchitecture;
    final elf = os == OS.linux || os == OS.android;
    final root = input.packageRoot.toFilePath();

    final sources = [
      for (final f in Directory.fromUri(
        input.packageRoot.resolve('src/'),
      ).listSync(recursive: true).whereType<File>())
        if (f.path.endsWith('.c')) f.path.substring(root.length),
    ]..sort();

    // Export the rayo_ functions only.
    String? versionScript;
    if (elf) {
      final file = File.fromUri(input.outputDirectory.resolve('rayo.map'));
      await file.writeAsString(
        '{\n  global:\n    rayo_*;\n  local:\n    *;\n};\n',
      );
      versionScript = file.path;
    }

    await CBuilder.library(
      name: 'rayo',
      assetName: 'src/bindings/rayo_bindings.dart',
      sources: sources,
      includes: ['include', 'src'],
      std: 'c89',
      flags: [
        '-Wall',
        '-Wextra',
        '-pedantic',
        '-Wdeclaration-after-statement',
        '-Wvla',
        '-Wconversion',
        '-Wshadow',
        '-fvisibility=hidden',
        '-ffunction-sections',
        '-fdata-sections',
        // Kernels for the target's SIMD. The dot-product kernels (AVX-VNNI,
        // ARM SDOT) are compiled with target attributes and chosen at run
        // time, so the baseline stays what every phone of the ABI runs.
        ...switch ((os, cpu)) {
          (OS.linux, Architecture.x64) => ['-mavx2', '-mfma'],
          (OS.android, Architecture.x64) => ['-msse4.2'],
          (OS.android, Architecture.arm) => ['-mfpu=neon'],
          _ => const <String>[],
        },
        // Reproducible: no build paths in the binary.
        '-ffile-prefix-map=$root=.',
        if (elf) ...[
          '-Wl,--gc-sections',
          '-Wl,--no-undefined',
          '-Wl,-z,relro,-z,now,-z,noexecstack',
          '-Wl,--build-id=none',
          '-Wl,--version-script=$versionScript',
        ],
        if (os == OS.iOS || os == OS.macOS) ...[
          '-Wl,-dead_strip',
          '-install_name',
          '@rpath/librayo.dylib',
        ],
      ],
      libraries: [if (elf) 'm'],
      linkModePreference: LinkModePreference.dynamic,
    ).run(input: input, output: output);
  });
}
