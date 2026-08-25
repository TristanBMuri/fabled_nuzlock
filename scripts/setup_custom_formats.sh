#!/bin/bash
# Register the no-gimmicks battle format for gauntlet training.
# Writes pokemon-showdown/config/custom-formats.ts, then servers must be restarted:
#   pkill -f "pokemon-showdown start" ; then restart on 8000-8023 (see README)
set -euo pipefail

SHOWDOWN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../pokemon-showdown" && pwd)"
CONFIG_DIR="$SHOWDOWN_DIR/config"
mkdir -p "$CONFIG_DIR"

cat > "$CONFIG_DIR/custom-formats.ts" << 'FORMATS_EOF'
// Custom formats for gauntlet training — no gimmicks.
// Note: no '!Sleep Clause Mod' — the base Custom Game format doesn't include
// Sleep Clause Mod, and trying to remove it crashes the server.

export const Formats: import('../sim/dex-formats').FormatList = [
	{
		section: "Custom",
	},
	{
		name: "[Gen 8] Custom Game (No Gimmicks)",
		mod: 'gen8',
		searchShow: true,
		challengeShow: true,
		battle: { trunc: Math.trunc },
		ruleset: [
			'Cancel Mod',
			'Max Team Size = 24', 'Max Move Count = 24',
			'Max Level = 9999', 'Default Level = 100',
			'Dynamax Clause',
		],
	},
];
FORMATS_EOF

echo "Created $CONFIG_DIR/custom-formats.ts"
echo "Restart Showdown servers to pick it up."
