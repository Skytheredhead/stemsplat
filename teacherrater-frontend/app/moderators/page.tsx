"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { X } from "lucide-react";

import { apiFetch, ensureCsrfToken } from "@/lib/api";
import { loadToken, loadUser, saveToken, saveUser } from "@/lib/auth";
import { useSchool } from "@/components/SchoolProvider";
import { ModerationReview, UserProfile } from "@/lib/types";

function downloadCsv(rows: ModerationReview[]) {
  const header = [
    "review_id",
    "teacher_name",
    "user_email",
    "rating",
    "difficulty",
    "clarity",
    "comment",
    "created_at",
  ];
  const csvRows = [header.join(",")];
  rows.forEach((row) => {
    const values = [
      row.id,
      row.teacher_name,
      row.user_email,
      row.rating,
      row.difficulty,
      row.clarity,
      row.comment,
      new Date(row.created_at).toISOString(),
    ].map((value) => `"${String(value).replace(/"/g, '""')}"`);
    csvRows.push(values.join(","));
  });
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "unmoderated-reviews.csv";
  link.click();
  URL.revokeObjectURL(url);
}

export default function ModeratorsPage() {
  const { school } = useSchool();
  const [reviews, setReviews] = useState<ModerationReview[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(loadToken());
  const [profile, setProfile] = useState<UserProfile | null>(loadUser<UserProfile>());
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [isProfileLoading, setIsProfileLoading] = useState(false);

  const isModerator = profile?.is_moderator ?? false;

  async function loadReviews() {
    if (!token) {
      setError("Please log in as a moderator.");
      return;
    }
    try {
      const data = await apiFetch<ModerationReview[]>("/moderation/reviews", {
        school,
        token,
      });
      setReviews(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load moderation queue.");
    }
  }

  useEffect(() => {
    const updateAuth = () => {
      setToken(loadToken());
      setProfile(loadUser<UserProfile>());
    };
    window.addEventListener("auth-updated", updateAuth);
    return () => window.removeEventListener("auth-updated", updateAuth);
  }, []);

  useEffect(() => {
    async function ensureProfile() {
      if (!token) {
        return;
      }
      if (!profile) {
        setIsProfileLoading(true);
        try {
          const data = await apiFetch<UserProfile>("/auth/me", { school, token });
          setProfile(data);
          saveUser(data);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Unable to validate session.");
        } finally {
          setIsProfileLoading(false);
        }
      }
    }
    ensureProfile();
  }, [school, token, profile]);

  useEffect(() => {
    if (!token || !isModerator) {
      return;
    }
    loadReviews();
  }, [school, token, isModerator]);

  async function handleModerate(reviewId: number) {
    if (!token) {
      return;
    }
    await ensureCsrfToken(school);
    await apiFetch(`/moderation/reviews/${reviewId}/moderate`, {
      method: "POST",
      school,
      token,
    });
    setReviews((prev) => prev.filter((review) => review.id !== reviewId));
  }

  async function handleDelete(reviewId: number) {
    if (!token) {
      return;
    }
    await ensureCsrfToken(school);
    await apiFetch(`/moderation/reviews/${reviewId}`, {
      method: "DELETE",
      school,
      token,
    });
    setReviews((prev) => prev.filter((review) => review.id !== reviewId));
  }

  const visibleRows = useMemo(() => reviews, [reviews]);

  if (!token) {
    return (
      <div className="relative min-h-[70vh]">
        <div className="fixed inset-0 -z-10 moderation-backdrop" aria-hidden="true" />
        <div className="flex min-h-[70vh] items-center justify-center">
          <div className="w-full max-w-lg rounded-3xl p-10 glass-panel-strong">
            <h2 className="text-center text-2xl font-semibold text-slate-900">
              moderation access
            </h2>
            <p className="mt-3 text-center text-sm text-slate-600">
              Test login: <span className="font-semibold">sussystudent26</span> · Password:{" "}
              <span className="font-semibold">sussystudent26</span>
            </p>
            <form
              className="mt-8 space-y-5"
              onSubmit={async (event) => {
                event.preventDefault();
                setError(null);
                setSubmitting(true);
                try {
                  await ensureCsrfToken(school);
                  const response = await apiFetch<{ access_token: string }>("/auth/login", {
                    method: "POST",
                    school,
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ identifier, password, honeypot: "" }),
                  });
                  saveToken(response.access_token);
                  setToken(response.access_token);
                  const userProfile = await apiFetch<UserProfile>("/auth/me", {
                    school,
                    token: response.access_token,
                  });
                  saveUser(userProfile);
                  setProfile(userProfile);
                  if (!userProfile.is_moderator) {
                    setError("This account does not have moderator access.");
                  }
                } catch (err) {
                  setError(err instanceof Error ? err.message : "Unable to authenticate.");
                } finally {
                  setSubmitting(false);
                }
              }}
            >
              <label className="block text-sm font-semibold text-slate-700">
                Username
                <input
                  type="text"
                  value={identifier}
                  onChange={(event) => setIdentifier(event.target.value)}
                  placeholder="Moderator username"
                  className="mt-2 w-full rounded-2xl px-5 py-4 text-lg text-slate-900 focus:outline-none glass-input"
                  required
                />
              </label>
              <label className="block text-sm font-semibold text-slate-700">
                Password
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Moderator password"
                  className="mt-2 w-full rounded-2xl px-5 py-4 text-lg text-slate-900 focus:outline-none glass-input"
                  required
                />
              </label>
              <button
                type="submit"
                disabled={submitting}
                className="w-full rounded-2xl border border-white/40 px-6 py-4 text-lg font-semibold text-white shadow-md transition glass-tint backdrop-blur-xl hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {submitting ? "Signing in..." : "Continue"}
              </button>
            </form>
            {error && <p className="mt-4 text-center text-sm text-red-600">{error}</p>}
          </div>
        </div>
      </div>
    );
  }

  if (!profile || isProfileLoading) {
    return (
      <div className="relative min-h-[70vh]">
        <div className="fixed inset-0 -z-10 moderation-backdrop" aria-hidden="true" />
        <div className="flex min-h-[70vh] items-center justify-center">
          <div className="w-full max-w-lg rounded-3xl p-10 text-center glass-panel-strong">
            <p className="text-sm text-slate-600">Checking moderator access…</p>
          </div>
        </div>
      </div>
    );
  }

  if (!isModerator) {
    return (
      <div className="relative min-h-[70vh]">
        <div className="fixed inset-0 -z-10 moderation-backdrop" aria-hidden="true" />
        <div className="space-y-3">
          <p className="text-sm text-slate-600">You do not have moderator access.</p>
          <Link
            href="/"
            className="inline-flex rounded-full border border-white/40 px-4 py-2 text-sm font-semibold text-white glass-tint shadow-md"
          >
            Return home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-[70vh]">
      <div className="fixed inset-0 -z-10 moderation-backdrop" aria-hidden="true" />
      <div className="space-y-6">
        <header className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900">Moderation queue</h2>
            <p className="text-sm text-slate-600">
              Review unmoderated comments for the selected school.
            </p>
          </div>
          <button
            onClick={() => downloadCsv(visibleRows)}
            className="rounded-full border border-white/40 px-4 py-2 text-sm font-semibold text-white shadow-md glass-tint"
            type="button"
          >
            Export CSV
          </button>
        </header>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <div className="space-y-4">
          {visibleRows.length === 0 && (
            <p className="text-sm text-slate-500">No unmoderated comments right now.</p>
          )}
          {visibleRows.map((review) => (
            <div
              key={review.id}
              className="flex flex-col gap-4 rounded-2xl p-5 glass-panel md:flex-row md:items-center md:justify-between"
            >
              <div>
                <p className="text-sm font-semibold text-slate-900">{review.teacher_name}</p>
                <p className="text-xs text-slate-500">{review.user_email}</p>
                <p className="mt-2 text-sm text-slate-700">{review.comment}</p>
                <p className="mt-2 text-xs text-slate-400">
                  {new Date(review.created_at).toLocaleString()}
                </p>
              </div>
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-slate-600">
                  <input
                    type="checkbox"
                    onChange={() => handleModerate(review.id)}
                    className="h-4 w-4 accent-theme-primary"
                  />
                  Mark moderated
                </label>
                <button
                  onClick={() => handleDelete(review.id)}
                  className="rounded-full border border-red-200/70 bg-red-500/20 px-3 py-1 text-sm font-semibold text-red-100 shadow-sm backdrop-blur-xl"
                  type="button"
                  aria-label="Delete review"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
