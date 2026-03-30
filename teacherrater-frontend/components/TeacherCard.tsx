import Link from "next/link";

import { Teacher } from "@/lib/types";

export function TeacherCard({ teacher }: { teacher: Teacher }) {
  return (
    <Link
      href={`/teachers/${teacher.id}`}
      className="group flex gap-4 rounded-2xl p-5 transition duration-200 glass-panel hover:scale-[1.02] hover:shadow-[0_0_30px_color-mix(in_srgb,var(--theme-primary)_35%,transparent)] active:scale-[0.98]"
    >
      <div className="overflow-hidden rounded-xl border border-white/60 glass-panel">
        <div className="flex h-28 w-20 items-center justify-center text-xs font-semibold text-slate-500 md:h-32 md:w-24">
          Photo
        </div>
      </div>
      <div className="flex-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-slate-900">{teacher.name}</h3>
          <span className="rounded-full border border-white/40 px-3 py-1 text-xs font-semibold text-white glass-tint">
            {teacher.subject}
          </span>
        </div>
        <p className="mt-2 text-sm text-slate-600">{teacher.department}</p>
      </div>
    </Link>
  );
}
