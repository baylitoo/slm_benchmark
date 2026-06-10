from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from docie_bench.schemas.common import DateField, MoneyField, NumberField, TextField

DynamicFieldType = Literal["string", "date", "number", "money"]
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RESERVED_FIELDS = {"document_type", "extraction_notes"}


class DynamicFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_FIELD_NAME_RE.pattern, description="Stable snake_case field name")
    type: DynamicFieldType
    description: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_name(self) -> DynamicFieldSpec:
        if not _FIELD_NAME_RE.fullmatch(self.name):
            raise ValueError("field name must be lower snake_case and at most 64 characters")
        if self.name in _RESERVED_FIELDS:
            raise ValueError(f"field name {self.name!r} is reserved")
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
    _PYDANTIC_TYPES: dict[DynamicFieldType, type[BaseModel]] = {
        "string": TextField,
        "date": DateField,
        "number": NumberField,
        "money": MoneyField,
    }
    _NUEXTRACT_TYPES: dict[DynamicFieldType, dict[str, str]] = {
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
            fields[field_spec.name] = (
                cls._PYDANTIC_TYPES[field_spec.type] | None,
                Field(default=None, description=field_spec.description),
            )
        model_name = (
            "".join(part.title() for part in spec.document_type.split("_")) + "DynamicExtraction"
        )
        return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)

    @classmethod
    def build_nuextract_template(cls, spec: DynamicSchemaSpec) -> dict[str, dict[str, str]]:
        return {field.name: dict(cls._NUEXTRACT_TYPES[field.type]) for field in spec.fields}
