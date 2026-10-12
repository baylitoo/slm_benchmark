# Manage extraction schemas in Studio

Open **Agents → Schemas** in the sidebar (or go to `/schemas`).
Saved schemas can be inspected, edited, duplicated under a new name, or deleted.
Built-in schemas are available for inspection and remain read-only.

Choose **New schema → Use resume starter** to begin with this structure:

```text
adbi_resume: {
  name: string
  title: string
  years_experience: number
  contact: {
    email: string
    phone: string
    linkedin: string
    github: string
    location: string
  }
  experience: list<{
    company: string
    title: string
    start_date: date
    end_date: date
    location: string
    description: string
    env_technique: string
  }>
  education: list<{
    degree: string
    institution: string
    year: string
  }>
  skills: list<{
    category: string
    items: list<{ item: string }>
  }>
  languages: list<{ language: string, level: string }>
  certifications: list<{ name: string, issuer: string, year: string }>
  interests: list<{ interest: string }>
}
```

A list defines the fields of **one item**, not the number of items to extract.
Define `company` and `title` once; the model can return zero, one, or many
experiences according to the document. The starter covers the ADBI resume field
structure, including grouped skills and mission-specific technical stacks. It
defines what to request; extraction completeness still depends on the model and
document. Existing saved schemas are not replaced when the starter changes.
Do not add numbered fields such as
`experience_1` and `experience_2`.

Use **Object (group of fields)** for a single nested group, or **List of objects**
for a repeated group. Both can contain further objects and lists, such as an
experience with its own list of projects. **Add child field** adds a property to
the item structure; it does not add an output item. The structure preview shows
the type declaration rather than a fixed-size example result.

Save the schema and select its name in Agents or Playground. Edits keep that
name and update the definition used by subsequent extractions. Past results
are not rewritten. Duplicate a schema before experimenting if existing agents
should continue using the original definition. Deleting a schema requires
confirmation; requests that still refer to its name must be updated.

The save, edit, and delete operations require a configured Studio database.
The edit endpoint is `PUT /v1/studio/schemas/dynamic/{name}`, accepting the same
`DynamicSchemaSpec` as creation. Its `document_type` must match the saved name.
Updates are in place (last committed edit wins); schema versioning is not part
of this change.
