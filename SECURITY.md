# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 3.4.x   | :white_check_mark: |
| < 3.4   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in MNN, please report it responsibly.

**DO NOT open a public GitHub issue for security vulnerabilities.**

Please email security reports to: **zhaode.wzd@alibaba-inc.com**

Include the following in your report:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Scope

The following are in scope for security reports:
- Memory safety issues (buffer overflow, use-after-free, etc.)
- Model file parsing vulnerabilities
- Input validation issues in inference APIs
- Vulnerabilities in the model converter

## Disclosure Policy

- We follow a 90-day coordinated disclosure timeline
- Security patches will be released as part of regular version updates
- Credit will be given to reporters in release notes (unless anonymity is requested)
