const fs = require("fs/promises");
const path = require("path");

/**
 * Recursively collects .sbn and .raw files grouped by directory.
 *
 * @param {string} rootDir - Directory to scan.
 * @returns {Promise<Array<{ sbn: string | null, raw: string | null }>>}
 */
async function collectSbnRawFiles(rootDir='/home/ai-developer/development/drs-to-fol/pmb/pmb-5.1.0/data/en/gold') {
  const groupedByDir = new Map();

  async function walk(currentDir) {
    const entries = await fs.readdir(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        await walk(fullPath);
        continue;
      }

      if (!entry.isFile()) continue;

      const lowerName = entry.name.toLowerCase();

      const isSbn = lowerName.includes(".sbn");
      const isRaw = lowerName.includes(".raw");

      if (!isSbn && !isRaw) continue;

      if (!groupedByDir.has(currentDir)) {
        groupedByDir.set(currentDir, {
          sbn: null,
          raw: null
        });
      }

      const group = groupedByDir.get(currentDir);
      const content = await fs.readFile(fullPath, "utf8");

      if (isSbn) {
        group.sbn = content.replace(/%[^\r\n]*/g, "");
      }

      if (isRaw) {
        group.raw = content;
      }
    }
  }

  await walk(rootDir);

  return Array.from(groupedByDir.values());
}

collectSbnRawFiles().then(data => fs.writeFile('gold.json', JSON.stringify(data)))