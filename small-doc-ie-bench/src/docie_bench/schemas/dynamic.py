from __future__ import annotations

import re
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


class DynamicTemplateBuilder:
    _PYDANTIC_TYPES: dict[ScalarDynamicFieldType, type[BaseModel]] = {
        "string": TextField,
        "date": DateField,
        "number": NumberField,
        "money": MoneyField,
    }
    _NUEXTRACT_TYPES: dict[ScalarDynamicFieldType, dict[str, str]] = {
        "string": {"value": "verbatim-string"},
        "date": {"value": "date"},
        "number": {"value": "number"},
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
            child.name: cls._model_field(child, parent_name=nested_name)
            for child in spec.fields
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
            return dict(cls._NUEXTRACT_TYPES[spec.type])
        nested = {child.name: cls._nuextract_field(child) for child in spec.fields}
        return [nested] if spec.type == "list" else nested
