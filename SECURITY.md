# Security Policy

## Supported Versions

This project is a competition submission for the Amazon ML Challenge 2026.
Security support is limited to the current `master` branch.

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please report it
responsibly.

### Preferred Method: GitHub Security Advisories

1. Go to the **Security** tab of this repository
2. Click **Report a vulnerability**
3. Fill in the details and submit

This allows for private disclosure and coordinated fix disclosure.

### Alternative: Private GitHub Issue

If Security Advisories are not available, open a **private issue**:

1. Go to **Issues** → **New issue**
2. Select **Security vulnerability** (if available) or mark as private
3. Describe the vulnerability without exposing exploit details

## What to Include

- Description of the vulnerability
- Steps to reproduce (minimal)
- Potential impact
- Suggested fix (if any)
- Your contact information for follow-up

## What NOT to Include

- **No API keys, secrets, tokens, or credentials** in reports
- **No exploit code** that could be used maliciously
- **No sensitive data** from the challenge dataset

## Response Timeline

- **Acknowledgment**: Within 7 days
- **Initial assessment**: Within 14 days
- **Fix timeline**: Depends on severity; critical issues prioritized

## Scope

This security policy covers:
- The Python package (`src/business_entity_resolution/`)
- Configuration and pipeline scripts
- Dependencies declared in `requirements.txt` / `pyproject.toml`

This policy does **not** cover:
- The Amazon ML Challenge infrastructure
- Third-party services or platforms
- User environments or deployments

## Responsible Disclosure

We appreciate responsible disclosure and will credit reporters (unless anonymity
is requested) in fix acknowledgments.

## No Private Security Email

This project does not maintain a dedicated security email address.
Please use GitHub's built-in security reporting mechanisms as described above.