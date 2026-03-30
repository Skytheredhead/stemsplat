"use client";

import { useEffect, useMemo, useState } from "react";
import { ListFilter, Search } from "lucide-react";

import { TeacherCard } from "@/components/TeacherCard";
import { apiFetch } from "@/lib/api";
import { useSchool } from "@/components/SchoolProvider";
import { SchoolKey, Teacher } from "@/lib/types";

const SCHOOL_OPTIONS: SchoolKey[] = ["sky", "iss", "lib"];

export default function HomePage() {
  const { school } = useSchool();
  const [teachersBySchool, setTeachersBySchool] = useState<Record<SchoolKey, Teacher[]>>({
    sky: [],
    iss: [],
    lib: [],
  });
  const [query, setQuery] = useState("");
  const [department, setDepartment] = useState("all");
  const [filterOpen, setFilterOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function loadTeachers() {
      try {
        const results = await Promise.all(
          SCHOOL_OPTIONS.map(async (schoolKey) => ({
            school: schoolKey,
            data: await apiFetch<Teacher[]>("/teachers", { school: schoolKey }),
          }))
        );
        if (!active) {
          return;
        }
        const nextTeachers: Record<SchoolKey, Teacher[]> = {
          sky: [],
          iss: [],
          lib: [],
        };
        results.forEach(({ school: schoolKey, data }) => {
          nextTeachers[schoolKey] = data;
        });
        setTeachersBySchool(nextTeachers);
        setError(null);
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Failed to load teachers");
        }
      }
    }
    loadTeachers();
    return () => {
      active = false;
    };
  }, []);

  const teachers = teachersBySchool[school] ?? [];

  const departments = useMemo(() => {
    const options = new Set(teachers.map((teacher) => teacher.department));
    return ["all", ...Array.from(options)];
  }, [teachers]);

  const filteredTeachers = useMemo(() => {
    const normalizedQuery = query.toLowerCase();
    return teachers
      .filter((teacher) => {
        const matchesQuery = teacher.name.toLowerCase().includes(normalizedQuery);
        const matchesDepartment = department === "all" || teacher.department === department;
        return matchesQuery && matchesDepartment;
      })
      .sort((a, b) => {
        if (department !== "all") {
          return a.name.localeCompare(b.name);
        }
        const deptCompare = a.department.localeCompare(b.department);
        if (deptCompare !== 0) {
          return deptCompare;
        }
        return a.name.localeCompare(b.name);
      });
  }, [teachers, query, department]);

  return (
    <div className="space-y-6">
      <div className="flex w-full justify-center px-4">
        <div className="flex w-full max-w-2xl flex-wrap items-center gap-3">
          <div className="relative">
            <button
              type="button"
              onClick={() => setFilterOpen((open) => !open)}
              className="flex h-12 w-12 items-center justify-center rounded-full text-slate-700 transition glass-panel hover:bg-white/50"
              aria-label="Filter departments"
            >
              <ListFilter className="h-5 w-5" />
            </button>
            {filterOpen && (
              <div className="absolute left-0 mt-3 w-56 overflow-hidden rounded-2xl shadow-xl glass-panel">
                {departments.map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => {
                      setDepartment(option);
                      setFilterOpen(false);
                    }}
                    className={`flex w-full items-center justify-between px-4 py-3 text-left text-sm capitalize transition ${
                      department === option
                        ? "bg-white/60 text-slate-900"
                        : "text-slate-700 hover:bg-white/40"
                    }`}
                  >
                    <span>{option === "all" ? "All departments" : option}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-4 top-1/2 z-10 h-5 w-5 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="find a teacher"
              className="relative z-0 h-12 w-full rounded-full pl-12 pr-4 text-sm text-slate-700 shadow-lg placeholder:text-slate-500 focus:border-white/80 focus:outline-none glass-input"
            />
          </div>
        </div>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <section className="mt-6 grid gap-4 sm:grid-cols-2">
        {filteredTeachers.map((teacher) => (
          <TeacherCard key={teacher.id} teacher={teacher} />
        ))}
      </section>
    </div>
  );
}
