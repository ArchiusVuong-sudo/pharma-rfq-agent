#!/bin/sh
# Starts the Claude Code CLI with an allowlisted environment.
# The Agent SDK merges the parent process environment into the CLI's; this wrapper
# drops all of it except the variables below (CLAUDECODE and CLAUDE_CODE_* are not passed).
# CLAUDE_AGENT_SDK_VERSION is set by the SDK itself for the child.
if [ -z "$RFQ_AGENT_REAL_CLI" ]; then
  echo "RFQ_AGENT_REAL_CLI is not set" >&2
  exit 2
fi
exec /usr/bin/env -i \
  PATH="$PATH" HOME="$HOME" USER="$USER" LANG="${LANG:-en_US.UTF-8}" TMPDIR="${TMPDIR:-/tmp}" \
  ${ANTHROPIC_API_KEY:+"ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"} \
  CLAUDE_AGENT_SDK_VERSION="$CLAUDE_AGENT_SDK_VERSION" \
  "$RFQ_AGENT_REAL_CLI" "$@"
