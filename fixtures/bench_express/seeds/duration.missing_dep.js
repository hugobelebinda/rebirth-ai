/**
 * Intentionally depends on the `ms` package which is NOT in package.json.
 * Phase 7 proves DependencyMissing → install_dependency → re-test.
 */
import ms from 'ms';

export function formatDuration(msValue) {
  return ms(msValue);
}
