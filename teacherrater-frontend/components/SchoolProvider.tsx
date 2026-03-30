"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

import { loadSchool, persistSchool, schoolThemes } from "@/lib/school";
import { SchoolKey } from "@/lib/types";

interface SchoolContextValue {
  school: SchoolKey;
  setSchool: (school: SchoolKey) => void;
}

const SchoolContext = createContext<SchoolContextValue | undefined>(undefined);

export function SchoolProvider({ children }: { children: React.ReactNode }) {
  const [school, setSchoolState] = useState<SchoolKey>("sky");

  useEffect(() => {
    const initial = loadSchool();
    setSchoolState(initial);
  }, []);

  useEffect(() => {
    persistSchool(school);
    const theme = schoolThemes[school].theme;
    document.documentElement.setAttribute("data-theme", theme);
  }, [school]);

  const value = useMemo(
    () => ({
      school,
      setSchool: setSchoolState,
    }),
    [school]
  );

  return <SchoolContext.Provider value={value}>{children}</SchoolContext.Provider>;
}

export function useSchool() {
  const context = useContext(SchoolContext);
  if (!context) {
    throw new Error("useSchool must be used within SchoolProvider");
  }
  return context;
}
