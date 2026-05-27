# Pull Request Rules

Pull Requests are welcome, but they must follow these rules:

## Required for Every Pull Request

Contributors must clearly describe:

* What was changed
* Why the change was made
* Which subsystem or module was affected
* Any architectural impact
* Performance or reasoning implications (if applicable)

Low-effort or unexplained Pull Requests may be rejected.

---

## Protected Core Files

Pull Requests that remove, replace, or intentionally damage important project files will not be accepted.

This includes files such as:

* `LICENSE`
* `README.md`
* `CONTRIBUTING.md`
* Core architecture files
* Research documentation
* Attribution or copyright notices

---

## Architectural Integrity

QMind is a research-oriented cognitive architecture.

Pull Requests that:

* Reduce explainability
* Convert the system into a black-box workflow
* Remove transparency mechanisms
* Add unnecessary complexity
* Conflict with the project's direction

may be rejected even if technically functional.

---

## Independent Forks Are Encouraged

If you have a different vision or experimental direction, you are encouraged to maintain your own fork instead of forcing incompatible architectural changes into the main branch.

Diverse experimentation is part of open-source research.
