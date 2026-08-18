import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const web = path.join(root, 'web');
const htmlNames = ['index.html', 'mobile.html', 'lan_login.html', 'install.html'];
const failures = [];

for (const name of htmlNames) {
  const source = fs.readFileSync(path.join(web, name), 'utf8');
  if (/<script(?!\s+src=)[^>]*>/i.test(source)) failures.push(`${name}: executable inline script`);
  if (/\son[a-z]+\s*=/i.test(source)) failures.push(`${name}: inline event handler`);
  if (/fonts\.googleapis\.com|fonts\.gstatic\.com|cdn\.tailwindcss\.com/i.test(source)) {
    failures.push(`${name}: runtime CDN dependency`);
  }
}

for (const asset of ['app.css', 'app.js', 'mobile.js', 'index-boot.js', 'mobile-boot.js', 'lan-login.js', 'install.js']) {
  if (!fs.existsSync(path.join(web, asset))) failures.push(`missing packaged asset: ${asset}`);
}

if (failures.length) {
  process.stderr.write(`${failures.join('\n')}\n`);
  process.exit(2);
}
process.stdout.write('offline frontend check passed\n');
