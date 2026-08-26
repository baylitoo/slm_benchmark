import { describe, expect, it } from "vitest";
import { detailOf } from "./core";

describe("detailOf", () => {
  it("returns a plain string detail as-is", () => {
    expect(detailOf({ detail: "batch has no failed items to retry" }, "fallback")).toBe(
      "batch has no failed items to retry",
    );
  });

  it("formats a FastAPI/pydantic 422 validation-error list into readable lines", () => {
    const body = {
      detail: [
        {
          type: "string_pattern_mismatch",
          loc: ["body", "document_type"],
          msg: "String should match pattern '^[a-z][a-z0-9_]{0,63}$'",
        },
      ],
    };
    expect(detailOf(body, "fallback")).toBe(
      "document_type: String should match pattern '^[a-z][a-z0-9_]{0,63}$'",
    );
  });

  it("joins multiple validation errors and drops the leading 'body' location segment", () => {
    const body = {
      detail: [
        { loc: ["body", "fields", 0, "name"], msg: "field required" },
        { loc: ["body", "document_type"], msg: "value is not a valid string" },
      ],
    };
    expect(detailOf(body, "fallback")).toBe(
      "fields.0.name: field required; document_type: value is not a valid string",
    );
  });

  it("falls back to fallback text when there's no usable detail", () => {
    expect(detailOf({}, "fallback")).toBe("fallback");
    expect(detailOf(null, "fallback")).toBe("fallback");
    expect(detailOf({ detail: [] }, "fallback")).toBe("[]");
  });
});
