import js from '@eslint/js';
import globals from 'globals';

export default [
  {
    ignores: ['node_modules/**', 'web/app.css'],
  },
  {
    files: ['web/*.js', 'tests/e2e/*.mjs', 'tools/*.mjs'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.node },
    },
    rules: {
      ...js.configs.recommended.rules,
      'no-undef': 'off',
      'no-unused-vars': 'off',
      'no-empty': 'off',
      'no-eval': 'error',
      'no-implied-eval': 'error',
    },
  },
];
