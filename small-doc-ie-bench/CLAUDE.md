# Project instructions

## Milestone-scale work: milestone → issues → integration branch → PRs → merge

When a body of work is genuinely milestone-scale — a real initiative spanning
several independently-shippable pieces working toward one larger capability,
not a single fix or feature — structure it like this, in order:

1. **Validate the scope before creating anything.** Propose the pillars/items
   as a plan in chat first (be creative here: this is where you weigh in with
   your own ideas, not just transcribe the user's list verbatim — but always
   let the user pick/cut/reshape before anything gets created). Don't create
   a GitHub milestone or issues until the user has actually confirmed scope.
2. **Create one GitHub milestone** for the initiative, with a description
   that states its pillars/goals and names the integration branch (step 3)
   so anyone reading the milestone later knows where the work actually lives.
3. **Create one integration branch off `master`**, e.g. `dev-<initiative>`.
   Every feature branch for this milestone branches off it and its PR
   targets it — not `master` directly. This is a **deliberate** exception to
   the normal "PR straight to master" flow, not the accidental stacked-PR
   trap (never merge a PR whose base isn't merged yet by mistake — this is
   the opposite: an intentional, named, shared base every PR in the
   milestone is supposed to build on).
4. **Break the scope into well-scoped GitHub issues**, one per
   independently-reviewable deliverable, each attached to the milestone.
   Note real dependencies between them (`Depends on #N`) so build order is
   visible. Prefer several small issues over one large one — same reasoning
   as the small-diff PR discipline elsewhere: each issue should be
   reviewable and mergeable on its own.
5. **Creating the milestone/issues/branch is scaffolding, not a start
   signal.** It does not, by itself, authorize building anything. Wait for
   an explicit go-ahead before opening the first feature branch/PR — the
   same "confirm before proceeding" discipline that applies to any
   scope-defining action, not a blanket permission slip because a milestone
   now exists.
6. **Build one issue at a time**, each as its own feature branch + PR
   against the integration branch (not `master`), following whatever the
   project's normal PR discipline is (tests, lint/type-check, PR body
   conventions, etc.) — the only difference from a normal PR is the base.
7. **When the milestone's scoped set is done**, merge the integration branch
   into `master` and close the milestone. Verify the merge against the
   actual tree (`git ls-tree`/`merge-base --is-ancestor`), never a status
   badge — the same discipline that applies to any "is this merged?"
   question.

The point of this shape: big initiatives get the same rigor as small
changes (scoped, reviewable, reversible units) without losing the
connective tissue between them (the milestone ties the pieces to the goal;
the integration branch lets them land incrementally without destabilizing
`master` mid-initiative). Follow the process precisely — but the scoping
and ideation inside it is exactly where judgment and creativity belong;
don't let the mechanism become an excuse to stop thinking about what's
actually worth building.
