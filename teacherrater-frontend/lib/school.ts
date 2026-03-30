import { SchoolKey } from "./types";

const SCHOOL_STORAGE_KEY = "ids-teachers-school";

export const schoolThemes: Record<SchoolKey, { label: string; theme: string }> = {
  sky: { label: "sky", theme: "sky" },
  iss: { label: "iss", theme: "iss" },
  lib: { label: "lib", theme: "lib" },
};

export function loadSchool(): SchoolKey {
  if (typeof window === "undefined") {
    return "sky";
  }
  const stored = window.localStorage.getItem(SCHOOL_STORAGE_KEY) as SchoolKey | null;
  return stored ?? "sky";
}

export function persistSchool(school: SchoolKey): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(SCHOOL_STORAGE_KEY, school);
}
