import { beforeEach, describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nProvider, useI18n } from "@/lib/i18n";
import { Button, Field, TextInput } from "./ui";
import { LanguageToggle } from "./LanguageToggle";

function LocaleProbe() {
  const { locale } = useI18n();
  return <output data-testid="locale">{locale}</output>;
}

function TestSurface() {
  return (
    <I18nProvider>
      <LanguageToggle />
      <LocaleProbe />
      <Field label="Document text" hint="Optional ISO code.">
        <TextInput placeholder="Paste the raw document text here…" />
      </Field>
      <Button>Save</Button>
    </I18nProvider>
  );
}

describe("LanguageToggle", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.lang = "en";
  });

  it("switches the studio to French and persists the preference", async () => {
    const user = userEvent.setup();
    render(<TestSurface />);

    await user.click(screen.getByRole("button", { name: "Switch to French" }));

    expect(screen.getByTestId("locale")).toHaveTextContent("fr");
    expect(screen.getByText("Texte du document")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Collez ici le texte brut du document…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enregistrer" })).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("fr");
    expect(window.localStorage.getItem("docie-studio-locale")).toBe("fr");
  });

  it("restores a saved French preference after hydration", async () => {
    window.localStorage.setItem("docie-studio-locale", "fr");
    render(<TestSurface />);

    await waitFor(() => expect(screen.getByTestId("locale")).toHaveTextContent("fr"));
    expect(screen.getByRole("button", { name: "Passer en anglais" })).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("fr");
  });
});
