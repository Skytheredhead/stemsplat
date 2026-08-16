import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const e2eHome = mkdtempSync(join(tmpdir(), 'stemsplat-playwright-'));
const executable = join(repositoryRoot, 'node_modules', '.bin', 'playwright');
const child = spawn(executable, ['test', ...process.argv.slice(2)], {
  cwd: repositoryRoot,
  env: { ...process.env, STEMSPLAT_E2E_HOME: e2eHome },
  stdio: 'inherit',
});

let exitCode = 1;
try {
  exitCode = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('close', (code, signal) => resolve(signal ? 1 : (code ?? 1)));
  });
} finally {
  rmSync(e2eHome, { recursive: true, force: true });
}
process.exitCode = exitCode;
