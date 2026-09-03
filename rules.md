# AI Agent Rules & Coding Standards

This file defines the coding rules and standards for this project. These rules apply to all contributors and AI agents (Cursor, Claude, Antigravity, Gemini, etc.) working on this codebase.

When writing, refactoring, or modifying code, you must adhere strictly to the following guidelines:

## 1. Variables and Mutability
- Prefer `const` over `let`. Use `let` only when explicitly planning to reassign.
- Never use `var`.
- Prefer immutable data patterns; avoid mutating input parameters.

## 2. Functions and Control Flow
- **Arrow Functions**: Use arrow functions only when the function body consists of 1 to 2 statements. For longer functions, use standard function declarations.
- **Function Design**: Keep functions short and focused (ideally under 30 lines). Prefer pure functions in utility modules. Name functions clearly using verb-noun conventions. Avoid boolean flag parameters; split into separate functions.
- **Early Returns**: Guard clauses should be at the top of the function. Prefer `if (condition) return;` over wrapping the entire function body in a conditional block. Keep the happy path unnested.

## 3. Arrays and Iterables
- Prefer `map()`, `filter()`, and `reduce()` over imperative `for` loops.
- Use `for...of` loops only when asynchronous sequential operations are required.
- Avoid `forEach()` when the return value matters; use `map()` instead.

## 4. Return Statements
- Avoid creating intermediate variables solely to return them immediately (e.g., `return db.query();` instead of `const result = await db.query(); return result;`).
- *Exception*: Assign to a variable first only when transformation or logging is needed before returning.

## 5. Types, Interfaces, and TypeScript Safety
- **Strict Typing**: Always declare explicit return types for exported functions and class methods.
- **Avoid `any`**: Never use `any`. Use specific interfaces, generics, or `unknown` (with type narrowing) instead.
- **Modern Operators**: Use Optional Chaining (`?.`) and Nullish Coalescing (`??`) for safe access to nullable or undefined values. Avoid non-null assertions (`!`) unless guaranteed, and document why.
- **Naming Interfaces**: All TypeScript interfaces must end with the `Interface` keyword (e.g., `UserInterface`).
- **Interface Locations**: Interfaces must live in the `interfaces` directory and use the `.interface.ts` file extension. *Exception*: Interfaces and types for `utils` or `lib` modules must be kept in the same file as the module they describe.
- **Colocation**: Define string and numeric constants used across an interface's domain in the same interface file. Export string union types as named `type` aliases.
- Prefer `interface` over `type` for object shapes. Use `type` for unions, intersections, and aliases.

## 6. Imports and Exports
- **Absolute Paths**: Always prefer absolute imports using the paths defined in `tsconfig.json` (e.g., `"lib/*"`, `"matomo/*"`). Avoid relative imports.
- **Import Ordering**: Group imports in this order: external libraries → internal `lib` modules → internal `matomo` modules → sibling modules.
- **Line Length**: Import statements must not exceed 80 characters. If they do, break the imports onto multiple lines with one item per line.
- **Named Exports**: Always use named exports. Never use `default` exports.

## 7. Strings and Formatting
- **Quotes**: Use double quotes (`"`) for all string literals. *Exception*: Use single quotes only to avoid escaping double quotes within the string (e.g., `'say "hello"'`).
- Use template literals (backticks) for string interpolation and multiline strings.
- **Destructuring**: Use object and array destructuring when pulling multiple properties out of an object.

## 8. Constants and Magic Values
- **No Magic Strings/Numbers**: Extract hardcoded, reusable values into clearly named constants at the top of the file.
- **Naming**: Use `UPPER_SNAKE_CASE` for all module-level constants.
- **Colocation**: Domain-wide string/numeric constants tied to an interface or type must be defined in the corresponding interface file.

## 9. Asynchronous Programming
- **Prefer Async/Await over Promises**: Always use `async/await` instead of `.then().catch()` chains or callbacks.
- **Parallel Execution**: Use `Promise.all()` when executing multiple independent async operations rather than awaiting them sequentially in a loop.

## 10. Logging and Error Handling
- **Structured Logging**: Avoid using `console.log` directly. Use the logger utility (`logger.info`, `logger.error`, etc.).
- **Log Formatting**: Every log statement must start with the name of the enclosing function, followed by a dash separator: `logger.info("[functionName] - message")`.
- **Custom Errors**: Throw standard `Error` objects or custom error classes rather than plain strings.
- **No Silent Catches**: Never leave a `catch` block completely empty. Always log errors with full context and handle them gracefully.
- **Side Effects**: Wrap side-effecting operations (DB, email, file I/O) in `try/catch` at the service layer. Propagate errors upward from data access and helpers.

## 11. Architecture and SOLID Principles
- **Strict 4-Layer Architecture**:
  - `dataAccess/`: Raw SQL queries only. No business logic.
  - `helpers/`: Business logic and orchestration. Delegates to data access.
  - `services/`: Job workflows, coordination, and side effects.
  - `utils/`: Pure, stateless helper functions.
  *Never skip layers or mix concerns.*
- **Single Responsibility (SRP)**: Each file, class, and function must have exactly one reason to change.
- **Open/Closed (OCP)**: Modules should be open for extension but closed for modification.
- **Liskov Substitution (LSP)**: Subtypes must be replaceable for their declared types without breaking consumers.
- **Interface Segregation (ISP)**: Create focused, narrow interfaces rather than monolithic object types.
- **Dependency Inversion (DIP)**: High-level modules must not depend on low-level modules directly. Inject dependencies.

## 12. File Naming Conventions
- Data Access: `*.dataAccess.ts`
- Helper: `*.helper.ts`
- Service: `*.service.ts`
- Interface: `*.interface.ts`
- Utility: `*.ts` (in `utils/`)

## 13. Comments and Documentation
- All exported functions must have a JSDoc comment describing: purpose, `@param`, and `@returns`.
- Inline comments should explain *why*, not *what* the code does.
- Remove commented-out code before merging.