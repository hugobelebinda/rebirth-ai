/**
 * Intentionally depends on the `pretty-ms` package, which is NOT in package.json, not imported
 * by the original src/duration.js, and not in express's dependency tree (unlike `ms`).
 * Phase 7 / 19: DependencyMissing → the package policy refuses it (bench_express) or, with
 * REBIRTH_ALLOW_UNDECLARED_PACKAGES=1, installs it in the setup step (bench_express_optin).
 */
import prettyMilliseconds from 'pretty-ms';

export function formatDuration(msValue) {
  return prettyMilliseconds(msValue);
}
