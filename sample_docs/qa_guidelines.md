# ACME Corp Software Quality Assurance (QA) Guidelines

This document details our engineering quality standards, testing expectations, and software release criteria.

## 1. Peer Code Review Requirements
All code changes must undergo a peer code review before being merged into the default branch of any repository. Diffs must be reviewed and approved by at least two senior engineers. Reviewers check for architectural compliance, security vulnerabilities, edge case coverage, and clear documentation.

## 2. Automated Test Coverage
We enforce a strict automated test coverage standard of at least 80% line coverage for all new services and features. Unit tests must be written for all business logic, while integration and API-level tests are required for endpoint routes. CI/CD pipelines will fail if test coverage falls below the 80% threshold.

## 3. Pre-Release Smoke and Regression Testing
Prior to major releases, QA engineers execute a comprehensive smoke test suite and regression testing on the staging environment. Smoke tests check the vital health paths of the application, including authentication, database connections, and third-party integrations, ensuring a stable release.

## 4. Defect Severity and Resolution Timeframes
Bugs identified in production are categorized into four severity tiers:
- **Severity 1 (Blocker)**: Critical service down. Must be resolved within 4 hours.
- **Severity 2 (High)**: Major feature broken with no workaround. Must be resolved within 24 hours.
- **Severity 3 (Medium)**: Feature degraded but functional with workaround. Must be resolved within 7 days.
- **Severity 4 (Low)**: Minor UI cosmetic or display issue. Resolved in standard release cycles.
