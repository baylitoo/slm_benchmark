from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from docie_bench.schemas.common import DateField, MoneyField, NumberField, TextField

DynamicFieldType = Literal["string", "date", "number", "money", "object", "list"]
ScalarDynamicFieldType = Literal["string", "date", "number", "money"]
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RESERVED_FIELDS = {"document_type", "extraction_notes"}


class DynamicFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_FIELD_NAME_RE.pattern, description="Stable snake_case field name")
    type: DynamicFieldType
    description: str | None = Field(default=None, max_length=300)
    fields: list[DynamicFieldSpec] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def validate_name(self) -> DynamicFieldSpec:
        if not _FIELD_NAME_RE.fullmatch(self.name):
            raise ValueError("field name must be lower snake_case and at most 64 characters")
        if self.name in _RESERVED_FIELDS:
            raise ValueError(f"field name {self.name!r} is reserved")
        if self.type in {"object", "list"} and not self.fields:
            raise ValueError(f"{self.type} field {self.name!r} must define nested fields")
        if self.type not in {"object", "list"} and self.fields:
            raise ValueError(f"scalar field {self.name!r} cannot define nested fields")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"nested field names in {self.name!r} must be unique")
        return self


class DynamicSchemaSpec(BaseModel):
    """Portable, validated description of a runtime extraction schema."""

    model_config = ConfigDict(extra="forbid")

    document_type: str = Field(min_length=1, max_length=64, pattern=_FIELD_NAME_RE.pattern)
    fields: list[DynamicFieldSpec] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def validate_schema(self) -> DynamicSchemaSpec:
        if not _FIELD_NAME_RE.fullmatch(self.document_type):
            raise ValueError("document_type must be lower snake_case and at most 64 characters")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("dynamic schema field names must be unique")
        return self


# How deep a request-supplied schema may nest. Real ones reach three or four
# levels; the ceiling exists so a hostile or generated one cannot exhaust the
# stack on the way in, which would reach the client as a 500 rather than a 400.
_MAX_SCHEMA_DEPTH = 24


_Step = tuple[str, bool]
"""One path segment: the key, and whether it holds a list."""


@dataclass(frozen=True)
class GliformerRecord:
    """One record type in the flat form, and where its values belong."""

    path: tuple[_Step, ...]
    fields: dict[str, tuple[_Step, ...]]
    leaves: list[tuple[_Step, ...]]


@dataclass(frozen=True)
class GliformerRecordPlan:
    """The flat record form sent to GLiFormer, and how to read the answer back.

    ``fields`` is what the model receives (``{"skills": ["category", "skill"]}``);
    ``records`` is the inverse, so reassembly looks up a path rather than
    guessing which record belongs where.
    """

    fields: dict[str, list[str]]
    records: dict[str, GliformerRecord]


def _name_paths(paths: list[tuple[_Step, ...]]) -> dict[str, tuple[_Step, ...]]:
    """Name each path by its last key, or by the whole path when that repeats.

    Two passes rather than first-come, so ``vendor.name`` never takes the name a
    top-level ``name`` also wants: a repeated leaf qualifies both.
    """
    repeated = {
        path[-1][0]
        for index, path in enumerate(paths)
        if any(other[-1][0] == path[-1][0] for other in paths[index + 1 :])
    }
    named: dict[str, tuple[_Step, ...]] = {}
    for path in paths:
        base = "_".join(key for key, _ in path) if path[-1][0] in repeated else path[-1][0]
        name, suffix = base, 2
        while name in named:
            name, suffix = f"{base}_{suffix}", suffix + 1
        named[name] = path
    return named


def gliformer_records_plan(
    schema: dict[str, Any], *, name: str = "record"
) -> GliformerRecordPlan:
    """Build GLiFormer's plain record form from a JSON Schema.

    ``{"employee": ["name", "company"]}`` is the form the model card documents
    first; :func:`gliformer_model_from_json_schema` builds the nested one.

    Every list becomes a record and every leaf below it becomes one of that
    record's fields, however deep — so a list inside a list contributes fields
    to its outermost list rather than a record of its own. That is what makes
    the answer reassemble: the model pairs the fields within a record, and no
    step has to guess which child row belongs to which parent.

    The depth and cycle guards of the nested builder apply here too: the schema
    arrives on a request.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}
    fields: dict[str, list[str]] = {}
    records: dict[str, GliformerRecord] = {}

    def deref(node: Any, followed: frozenset[str]) -> tuple[Any, frozenset[str]]:
        while isinstance(node, dict) and "$ref" in node:
            ref = str(node["$ref"]).rsplit("/", 1)[-1]
            if ref in followed:
                raise ValueError(f"schema definition {ref!r} refers to itself")
            followed = followed | {ref}
            node = defs.get(ref, {})
        return node, followed

    def pick(node: Any, followed: frozenset[str]) -> tuple[Any, frozenset[str]]:
        node, followed = deref(node, followed)
        if not isinstance(node, dict):
            return {}, followed
        for key in ("anyOf", "oneOf"):
            choices = node.get(key)
            if not isinstance(choices, list):
                continue
            for choice in choices:
                chosen, chosen_followed = deref(choice, followed)
                if isinstance(chosen, dict) and chosen.get("type") != "null":
                    return chosen, chosen_followed
        return node, followed

    def collect(
        node: Any,
        record: GliformerRecord,
        below: tuple[_Step, ...],
        depth: int,
        followed: frozenset[str],
    ) -> None:
        """Add ``node``'s leaves to ``record``, opening a record for each list."""
        if depth > _MAX_SCHEMA_DEPTH:
            raise ValueError(f"schema nests deeper than {_MAX_SCHEMA_DEPTH} levels")
        node, followed = pick(node, followed)
        properties = node.get("properties") if isinstance(node, dict) else None
        if not isinstance(properties, dict):
            return
        for key, value in properties.items():
            if key in _RESERVED_FIELDS:
                continue
            child, child_followed = pick(value, followed)
            is_list = isinstance(child, dict) and (
                child.get("type") == "array" or "items" in child
            )
            inner, inner_followed = (
                pick(child.get("items") or {}, child_followed)
                if is_list
                else (child, child_followed)
            )
            path = (*below, (key, is_list))
            has_properties = isinstance(inner, dict) and isinstance(
                inner.get("properties"), dict
            )
            if is_list and not record.path and has_properties:
                # A list with no list above it opens its own record.
                opened = GliformerRecord(path=path, fields={}, leaves=[])
                opened_records.append(opened)
                collect(inner, opened, (), depth + 1, inner_followed)
            elif has_properties:
                collect(inner, record, path, depth + 1, inner_followed)
            else:
                record.leaves.append(path)

    root = GliformerRecord(path=(), fields={}, leaves=[])
    opened_records: list[GliformerRecord] = []
    collect(schema, root, (), 0, frozenset())

    records[name] = root
    for named, opened in _name_paths([one.path for one in opened_records]).items():
        records[named if named != name else f"{named}_record"] = next(
            one for one in opened_records if one.path == opened
        )
    for record_name, record in records.items():
        record.fields.update(_name_paths(record.leaves))
        if record.fields:
            fields[record_name] = list(record.fields)
    if not fields:
        raise ValueError("schema has no fields to structure into")
    return GliformerRecordPlan(fields=fields, records=records)


def _place(container: dict[str, Any], path: tuple[_Step, ...], value: Any) -> None:
    """Write ``value`` at ``path``, opening one-element lists on the way down."""
    for index, (key, is_list) in enumerate(path):
        if index == len(path) - 1:
            container[key] = [value] if is_list else value
            return
        nested = container.get(key)
        if is_list:
            if not (isinstance(nested, list) and nested and isinstance(nested[0], dict)):
                nested = [{}]
                container[key] = nested
            container = nested[0]
        else:
            if not isinstance(nested, dict):
                nested = {}
                container[key] = nested
            container = nested


def document_from_gliformer_records(
    answer: Any, plan: GliformerRecordPlan, *, name: str = "record"
) -> dict[str, Any]:
    """Read the flat records back into the shape the schema describes.

    The exact inverse of :func:`gliformer_records_plan`: each field goes back to
    the path it was taken from. A list that held another list gets one parent
    row per item, because that is how the model returned them.
    """
    if not isinstance(answer, dict):
        return {}
    document: dict[str, Any] = {}
    for record_name, record in plan.records.items():
        raw = answer.get(record_name)
        rows = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
        rows = [row for row in rows if isinstance(row, dict)]
        if not record.path:
            # The root record carries the document's own fields, so a field
            # answered on more than one chunk keeps the first answer.
            answered: set[tuple[_Step, ...]] = set()
            for row in rows:
                for field, path in record.fields.items():
                    value = row.get(field)
                    if value in (None, "") or path in answered:
                        continue
                    _place(document, path, value)
                    answered.add(path)
            continue
        built: list[dict[str, Any]] = []
        for row in rows:
            out: dict[str, Any] = {}
            for field, path in record.fields.items():
                value = row.get(field)
                if value not in (None, ""):
                    _place(out, path, value)
            if out:
                built.append(out)
        if built:
            # The last step is the list itself, so it takes the rows as they are.
            _place(document, (*record.path[:-1], (record.path[-1][0], False)), built)
    return document


def gliformer_model_from_json_schema(
    schema: dict[str, Any], *, name: str = "Record"
) -> type[BaseModel]:
    """Build the Pydantic model GLiFormer's ``structure()`` takes from a JSON Schema.

    The extraction path already sends a schema on every request, as
    ``response_format.json_schema.schema``. This turns that into the one form
    GLiFormer accepts, so a served GLiFormer needs no schema channel of its own.

    Every leaf is ``str | None``, dates and amounts included: GLiFormer copies
    spans off the page, and typing them is the job of the pass that already runs
    after extraction (``normalize_by_schema`` then ``coerce_scalars``).
    Declaring a Decimal here would make ``validate_output=True`` reject
    ``"1 234,56 EUR"`` rather than accept it.

    The schema comes off a request, so two shapes are refused with ``ValueError``
    rather than followed: one nesting past :data:`_MAX_SCHEMA_DEPTH`, and a
    definition that refers back to itself. Both describe an infinitely deep
    record and both would otherwise raise ``RecursionError``, which the caller
    cannot turn into a 400.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def deref(node: Any, followed: frozenset[str]) -> tuple[Any, frozenset[str]]:
        """Follow ``$ref`` links, refusing one already followed on this path."""
        while isinstance(node, dict) and "$ref" in node:
            ref = str(node["$ref"]).rsplit("/", 1)[-1]
            if ref in followed:
                raise ValueError(f"schema definition {ref!r} refers to itself")
            followed = followed | {ref}
            node = defs.get(ref, {})
        return node, followed

    def pick(node: Any, followed: frozenset[str]) -> tuple[Any, frozenset[str]]:
        """The non-null branch of a union, dereferenced."""
        node, followed = deref(node, followed)
        if not isinstance(node, dict):
            return {}, followed
        for key in ("anyOf", "oneOf"):
            choices = node.get(key)
            if not isinstance(choices, list):
                continue
            for choice in choices:
                chosen, chosen_followed = deref(choice, followed)
                if isinstance(chosen, dict) and chosen.get("type") != "null":
                    return chosen, chosen_followed
        return node, followed

    def build(node: Any, label: str, depth: int, followed: frozenset[str]) -> Any:
        if depth > _MAX_SCHEMA_DEPTH:
            raise ValueError(f"schema nests deeper than {_MAX_SCHEMA_DEPTH} levels")
        node, followed = pick(node, followed)
        properties = node.get("properties") if isinstance(node, dict) else None
        if isinstance(properties, dict) and properties:
            fields = {
                key: (
                    build(
                        value,
                        f"{label}{key.title().replace('_', '')}",
                        depth + 1,
                        followed,
                    ),
                    Field(default=None),
                )
                for key, value in properties.items()
                if key not in _RESERVED_FIELDS
            }
            if not fields:
                return str | None
            return create_model(  # type: ignore[call-overload]
                label or "Record", __config__=ConfigDict(extra="forbid"), **fields
            )
        if isinstance(node, dict) and (node.get("type") == "array" or "items" in node):
            inner = build(node.get("items") or {}, label + "Item", depth + 1, followed)
            return list[inner] | None  # type: ignore[valid-type]
        return str | None

    label = "".join(part.title() for part in name.split("_")) or "Record"
    built = build(schema, label, 0, frozenset())
    if isinstance(built, type) and issubclass(built, BaseModel):
        return built
    raise ValueError("schema has no object properties to structure into")


class DynamicTemplateBuilder:
    _PYDANTIC_TYPES: dict[ScalarDynamicFieldType, type[BaseModel]] = {
        "string": TextField,
        "date": DateField,
        "number": NumberField,
        "money": MoneyField,
    }
    # Bare type strings, NuExtract's own template shape; money is a real
    # two-field object. See llm.prompts._NUEXTRACT_TEMPLATES.
    _NUEXTRACT_TYPES: dict[ScalarDynamicFieldType, str | dict[str, str]] = {
        "string": "verbatim-string",
        "date": "date",
        "number": "number",
        "money": {"amount": "number", "currency": "currency"},
    }

    @classmethod
    def build_model(cls, spec: DynamicSchemaSpec) -> type[BaseModel]:
        fields: dict[str, Any] = {
            "document_type": (Literal[spec.document_type], spec.document_type),
            "extraction_notes": (list[str], Field(default_factory=list)),
        }
        for field_spec in spec.fields:
            fields[field_spec.name] = cls._model_field(
                field_spec,
                parent_name="".join(part.title() for part in spec.document_type.split("_")),
            )
        model_name = (
            "".join(part.title() for part in spec.document_type.split("_")) + "DynamicExtraction"
        )
        return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)

    @classmethod
    def build_nuextract_template(cls, spec: DynamicSchemaSpec) -> dict[str, Any]:
        return {field.name: cls._nuextract_field(field) for field in spec.fields}

    @classmethod
    def build_gliformer_schema(cls, spec: DynamicSchemaSpec) -> dict[str, Any]:
        """The Pydantic form GLiFormer's ``structure()`` takes: ``{name: Model}``.

        Every leaf is ``str | None``, including dates and amounts: GLiFormer
        copies spans off the page, and turning ``"1 234,56 €"`` into a number is
        the job of the schema-typed pass the extraction pipeline already runs
        (``normalize_by_schema`` then ``coerce_scalars``). Declaring a Decimal
        here would make ``validate_output=True`` reject the span instead.
        """
        model = create_model(  # type: ignore[call-overload]
            "".join(part.title() for part in spec.document_type.split("_")) + "Gliformer",
            __config__=ConfigDict(extra="forbid"),
            **{field.name: cls._gliformer_field(field) for field in spec.fields},
        )
        return {spec.document_type: model}

    @classmethod
    def build_gliformer_fields(cls, spec: DynamicSchemaSpec) -> dict[str, list[str]]:
        """The plain form: ``{record: [field, ...]}``.

        Lighter than the Pydantic schema and all GLiFormer needs for a flat
        record. A nested field contributes its own record, keyed by field name,
        because this form cannot express nesting.
        """
        records: dict[str, list[str]] = {}

        def collect(name: str, fields: list[DynamicFieldSpec]) -> None:
            leaves: list[str] = []
            for field in fields:
                if field.type in {"object", "list"}:
                    collect(field.name, list(field.fields))
                else:
                    leaves.append(field.name)
            if leaves:
                records[name] = leaves

        collect(spec.document_type, list(spec.fields))
        return records

    @classmethod
    def _gliformer_field(cls, spec: DynamicFieldSpec) -> tuple[Any, Any]:
        field_info = Field(default=None, description=spec.description)
        if spec.type == "money":
            money = create_model(  # type: ignore[call-overload]
                "".join(part.title() for part in spec.name.split("_")) + "GliformerMoney",
                __config__=ConfigDict(extra="forbid"),
                amount=(str | None, Field(default=None)),
                currency=(str | None, Field(default=None)),
            )
            return money | None, field_info
        if spec.type not in {"object", "list"}:
            return str | None, field_info
        nested = create_model(  # type: ignore[call-overload]
            "".join(part.title() for part in spec.name.split("_")) + "Gliformer",
            __config__=ConfigDict(extra="forbid"),
            **{child.name: cls._gliformer_field(child) for child in spec.fields},
        )
        if spec.type == "list":
            return list[nested] | None, field_info  # type: ignore[valid-type]
        return nested | None, field_info

    @classmethod
    def _model_field(cls, spec: DynamicFieldSpec, *, parent_name: str) -> tuple[Any, Any]:
        field_info = Field(default=None, description=spec.description)
        if spec.type not in {"object", "list"}:
            return cls._PYDANTIC_TYPES[spec.type] | None, field_info

        nested_name = parent_name + "".join(part.title() for part in spec.name.split("_"))
        nested_fields = {
            child.name: cls._model_field(child, parent_name=nested_name) for child in spec.fields
        }
        nested_model = create_model(
            nested_name,
            __config__=ConfigDict(extra="forbid"),
            **nested_fields,
        )
        if spec.type == "list":
            return list[nested_model] | None, field_info
        return nested_model | None, field_info

    @classmethod
    def _nuextract_field(cls, spec: DynamicFieldSpec) -> Any:
        if spec.type not in {"object", "list"}:
            leaf = cls._NUEXTRACT_TYPES[spec.type]
            return dict(leaf) if isinstance(leaf, dict) else leaf
        nested = {child.name: cls._nuextract_field(child) for child in spec.fields}
        return [nested] if spec.type == "list" else nested
