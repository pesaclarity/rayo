import 'dart:isolate';

import 'session.dart';

/// Runs [task] on [session] in a short-lived isolate, so a UI isolate never
/// waits on a run. Runs on one session must not overlap: await each before
/// the next (or keep a queue).
Future<R> runInIsolate<R>(
  RayoSession session,
  R Function(RayoSession session) task,
) {
  final ref = session.reference;
  return Isolate.run(() {
    final borrowed = RayoSession.borrow(ref);
    try {
      return task(borrowed);
    } finally {
      borrowed.close();
    }
  });
}
