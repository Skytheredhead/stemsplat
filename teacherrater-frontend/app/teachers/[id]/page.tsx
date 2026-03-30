"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { ReviewCard } from "@/components/ReviewCard";
import { ReviewForm } from "@/components/ReviewForm";
import { apiFetch, ensureCsrfToken } from "@/lib/api";
import { loadToken } from "@/lib/auth";
import { useSchool } from "@/components/SchoolProvider";
import { Review, TeacherDetail } from "@/lib/types";

const sortOptions = [
  { value: "newest", label: "Newest" },
  { value: "highest", label: "Highest rating" },
  { value: "lowest", label: "Lowest rating" },
] as const;

type SortOption = (typeof sortOptions)[number]["value"];

export default function TeacherDetailPage({ params }: { params: { id: string } }) {
  const { school } = useSchool();
  const router = useRouter();
  const [detail, setDetail] = useState<TeacherDetail | null>(null);
  const [sortBy, setSortBy] = useState<SortOption>("newest");
  const [error, setError] = useState<string | null>(null);
  const token = loadToken();

  async function loadDetail() {
    try {
      const data = await apiFetch<TeacherDetail>(`/teachers/${params.id}`, { school });
      setDetail(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load teacher");
    }
  }

  useEffect(() => {
    loadDetail();
  }, [school, params.id]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  useEffect(() => {
    if (!detail) {
      return;
    }
    const hash = window.location.hash.replace("#", "");
    if (!hash) {
      return;
    }
    const target = document.getElementById(hash);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [detail]);

  const sortedReviews = useMemo(() => {
    if (!detail) {
      return [];
    }
    const reviews = [...detail.reviews];
    if (sortBy === "highest") {
      reviews.sort((a, b) => b.rating - a.rating);
    } else if (sortBy === "lowest") {
      reviews.sort((a, b) => a.rating - b.rating);
    } else {
      reviews.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    }
    return reviews;
  }, [detail, sortBy]);

  async function handleReport(review: Review) {
    if (!token) {
      setError("Please log in to report a review.");
      return;
    }
    const reason = window.prompt("Why are you reporting this review?");
    if (!reason) {
      return;
    }
    const allowedPattern = /^[A-Za-z0-9 ,.?!]+$/;
    if (!allowedPattern.test(reason) || reason.length > 500) {
      setError("Report reason has unsupported characters or is too long.");
      return;
    }
    try {
      await ensureCsrfToken(school);
      await apiFetch(`/reviews/${review.id}/reports`, {
        method: "POST",
        school,
        token,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      });
      setError("Report submitted. Thank you.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to report review.");
    }
  }

  if (!detail) {
    return <p className="text-sm text-slate-500">Loading...</p>;
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-slate-900/30 px-4 backdrop-blur-md"
      onClick={() => {
        router.push("/");
      }}
    >
      <div
        className="relative w-full max-w-5xl overflow-hidden rounded-3xl p-6 glass-panel-strong"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="max-h-[85vh] overflow-y-auto pr-2">
          <div className="space-y-6">
            <section className="rounded-2xl p-6 glass-panel">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div className="flex flex-col gap-4 md:flex-row md:items-center">
                  <div className="h-32 w-24 overflow-hidden rounded-2xl border border-white/60 glass-panel md:h-40 md:w-32">
                    <div className="flex h-full w-full items-center justify-center text-xs font-semibold text-slate-500">
                      Photo
                    </div>
                  </div>
                  <div>
                    <h2 className="text-2xl font-semibold text-slate-900">
                      {detail.teacher.name}
                    </h2>
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-slate-700">
                      <span className="rounded-full border border-white/40 px-3 py-1 text-xs font-semibold text-white glass-tint">
                        {detail.teacher.subject}
                      </span>
                      <span>{detail.teacher.department}</span>
                    </div>
                  </div>
                </div>
                <div className="rounded-xl border border-white/40 px-4 py-3 text-center text-white glass-tint">
                  <p className="text-sm text-white/80">Average rating</p>
                  <p className="text-2xl font-semibold">
                    {detail.average_rating ? detail.average_rating.toFixed(1) : "No reviews"}
                  </p>
                </div>
              </div>
            </section>

            <ReviewForm teacherId={detail.teacher.id} onSubmitted={loadDetail} />

            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <h3 className="text-xl font-semibold text-slate-900">Reviews</h3>
              <select
                value={sortBy}
                onChange={(event) => setSortBy(event.target.value as SortOption)}
                className="rounded-full px-4 py-2 text-sm glass-input"
              >
                {sortOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    Sort by {option.label}
                  </option>
                ))}
              </select>
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}

            <div className="space-y-4">
              {sortedReviews.length === 0 && (
                <p className="text-sm text-slate-600">No reviews yet. Be the first to post one.</p>
              )}
              {sortedReviews.map((review) => (
                <div key={review.id} className="space-y-2">
                  <ReviewCard review={review} />
                  <button
                    onClick={() => handleReport(review)}
                    className="text-xs font-semibold text-slate-700 hover:text-slate-900"
                    type="button"
                  >
                    Report review
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
