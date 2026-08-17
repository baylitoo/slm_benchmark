"use client";

import {
  Children,
  cloneElement,
  createContext,
  isValidElement,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { FR_MESSAGES, translateFrenchPattern } from "./i18n.fr";

export type Locale = "en" | "fr";

const STORAGE_KEY = "docie-studio-locale";

type Variables = Record<string, string | number>;

function interpolate(message: string, variables?: Variables): string {
  if (!variables) return message;
  return message.replace(/\{(\w+)\}/g, (match, name: string) =>
    Object.prototype.hasOwnProperty.call(variables, name)
      ? String(variables[name])
      : match,
  );
}

function translate(locale: Locale, message: string, variables?: Variables): string {
  const localized =
    locale === "fr" ? (FR_MESSAGES[message] ?? translateFrenchPattern(message)) : undefined;
  return interpolate(localized ?? message, variables);
}

interface I18nValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
  t: (message: string, variables?: Variables) => string;
}

const I18nContext = createContext<I18nValue>({
  locale: "en",
  setLocale: () => undefined,
  toggleLocale: () => undefined,
  t: (message, variables) => interpolate(message, variables),
});

export function I18nProvider({ children }: { children: ReactNode }) {
  // Always render English first so server and client markup agree. A saved
  // preference is applied immediately after hydration.
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "fr") setLocaleState(saved);
  }, []);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    window.localStorage.setItem(STORAGE_KEY, next);
  }, []);

  const toggleLocale = useCallback(() => {
    setLocaleState((current) => {
      const next = current === "en" ? "fr" : "en";
      window.localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const t = useCallback(
    (message: string, variables?: Variables) => translate(locale, message, variables),
    [locale],
  );

  const value = useMemo(
    () => ({ locale, setLocale, toggleLocale, t }),
    [locale, setLocale, toggleLocale, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  return useContext(I18nContext);
}

/** Translate string leaves while preserving icons and other React elements. */
export function useTranslatedNode() {
  const { t } = useI18n();
  return useCallback(
    function translateNode(node: ReactNode): ReactNode {
      if (typeof node === "string") return t(node);
      if (Array.isArray(node)) return node.map(translateNode);
      return node;
    },
    [t],
  );
}

/** Small escape hatch for prose that is rendered directly rather than by a UI primitive. */
export function T({ children }: { children: string | string[] }) {
  const { t } = useI18n();
  return <>{t(Array.isArray(children) ? children.join("") : children)}</>;
}

/** Translate option/optgroup trees passed to the shared Select primitive. */
export function useTranslatedOptions() {
  const { t } = useI18n();
  return useCallback(
    function translateOptions(node: ReactNode): ReactNode {
      return Children.map(node, (child) => {
        if (!isValidElement(child)) {
          return typeof child === "string" ? t(child) : child;
        }
        const element = child as ReactElement<{ children?: ReactNode; label?: string }>;
        const props: { children?: ReactNode; label?: string } = {};
        if (element.props.children !== undefined) {
          props.children = translateOptions(element.props.children);
        }
        if (typeof element.props.label === "string") props.label = t(element.props.label);
        return Object.keys(props).length ? cloneElement(element, props) : element;
      });
    },
    [t],
  );
}
