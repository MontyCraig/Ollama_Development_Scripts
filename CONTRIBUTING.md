# Contributing Guidelines

## Code Standards

* Follow PEP 8 style guide
* Add type hints to all functions
* Include docstrings for all modules, classes, and functions
* Write unit tests for new features
* Keep functions focused and under 50 lines
* Use meaningful variable and function names

## Commit Messages

Follow conventional commits:
* feat: New feature
* fix: Bug fix
* docs: Documentation only changes
* style: Changes that do not affect the meaning of the code
* refactor: Code change that neither fixes a bug nor adds a feature
* test: Adding missing tests
* chore: Changes to the build process or auxiliary tools

## Development Workflow

1. Create feature branch from development
   ```bash
   git checkout -b feature/your-feature-name development
   ```

2. Make changes and commit
   ```bash
   git add .
   git commit -m "feat: Your feature description"
   ```

3. Push changes
   ```bash
   git push origin feature/your-feature-name
   ```

4. Before creating PR:
   * Run all tests
   * Update documentation
   * Check code formatting
   * Resolve merge conflicts

5. Create Pull Request to development branch

6. After review and approval, merge to development

7. Clean up after merge:
   * Delete feature branch
   * Update local repository
   * Verify changes in development

6. Periodically merge development into main for releases 