import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';

const envPath = path.join(process.cwd(), '.env.example');
assert.ok(fs.existsSync(envPath), '.env.example should exist for frontend');

const contents = fs.readFileSync(envPath, 'utf8');
assert.ok(contents.includes('NEXT_PUBLIC_API_URL='), 'NEXT_PUBLIC_API_URL should be documented in .env.example');
assert.ok(contents.includes('API_URL='), 'API_URL should be documented in .env.example');
