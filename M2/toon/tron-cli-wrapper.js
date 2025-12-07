#!/usr/bin/env node

import { TRON } from '@tron-format/tron';
import fs from 'fs';

// Parse command line arguments
// Expected: node tron-cli-wrapper.js <input.json> -o <output.tron>

const args = process.argv.slice(2);
let inputFile, outputFile;

// Parse arguments
for (let i = 0; i < args.length; i++) {
  if (args[i] === '-o' && i + 1 < args.length) {
    outputFile = args[i + 1];
    i++;
  } else if (!inputFile) {
    inputFile = args[i];
  }
}

// Validation
if (!inputFile || !outputFile) {
  console.error('Usage: tron-cli-wrapper.js <input.json> -o <output.tron>');
  process.exit(1);
}

try {
  // Read JSON input
  const jsonData = fs.readFileSync(inputFile, 'utf-8');
  const data = JSON.parse(jsonData);

  // Convert to TRON format
  const tronString = TRON.stringify(data);

  // Write TRON output
  fs.writeFileSync(outputFile, tronString, 'utf-8');

  console.log(`Successfully converted ${inputFile} to TRON format`);
  process.exit(0);
} catch (error) {
  console.error(`Error: ${error.message}`);
  process.exit(1);
}
