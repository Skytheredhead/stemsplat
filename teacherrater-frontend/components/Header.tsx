"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Bell, BellDot } from "lucide-react";

import { AuthPanel } from "@/components/AuthPanel";
import { useSchool } from "@/components/SchoolProvider";
import { clearToken, loadToken, loadUser } from "@/lib/auth";
import { loadNotifications, removeNotification } from "@/lib/notifications";
import { schoolThemes } from "@/lib/school";
import { NotificationItem, SchoolKey, UserProfile } from "@/lib/types";

const schoolOptions: SchoolKey[] = ["sky", "iss", "lib"];

export function Header() {
  const { school, setSchool } = useSchool();
  const pathname = usePathname();
  const router = useRouter();
  const [username, setUsername] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [userId, setUserId] = useState<number | null>(null);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [authOpen, setAuthOpen] = useState(false);
  const [authSwap, setAuthSwap] = useState(false);

  const isTeacherPage = pathname.startsWith("/teachers/");
  const isModeratorPage = pathname.startsWith("/moderators");
  const isModeratorOnly = process.env.NEXT_PUBLIC_MODERATOR_ONLY === "true";
  const isHome = pathname === "/";
  const hasNotifications = notifications.length > 0;

  if (isModeratorPage || isModeratorOnly) {
    return null;
  }

  useEffect(() => {
    setAuthSwap(true);
    const timeout = window.setTimeout(() => setAuthSwap(false), 320);
    return () => window.clearTimeout(timeout);
  }, [token, username]);

  useEffect(() => {
    if (token) {
      setAuthOpen(false);
    }
  }, [token]);

  useEffect(() => {
    if (!userId || !token) {
      setNotifications([]);
      return;
    }
    let active = true;
    loadNotifications(school, token)
      .then((data) => {
        if (active) {
          setNotifications(data);
        }
      })
      .catch(() => {
        if (active) {
          setNotifications([]);
        }
      });
    return () => {
      active = false;
    };
  }, [userId, token, school]);

  useEffect(() => {
    if (!notificationsOpen || !userId || !token) {
      return;
    }
    loadNotifications(school, token)
      .then(setNotifications)
      .catch(() => setNotifications([]));
  }, [notificationsOpen, userId, token, school]);

  useEffect(() => {
    const updateAuth = () => {
      setToken(loadToken());
      const profile = loadUser<UserProfile>();
      setUsername(profile?.username ?? null);
      setUserId(profile?.id ?? null);
    };
    updateAuth();
    window.addEventListener("auth-updated", updateAuth);
    return () => window.removeEventListener("auth-updated", updateAuth);
  }, []);

  function handleLogout() {
    clearToken();
    setToken(null);
    setUsername(null);
    setUserId(null);
    setNotificationsOpen(false);
  }

  function handleSchoolClick(next: SchoolKey) {
    if (next === school) {
      return;
    }
    if (!isHome && isTeacherPage) {
      setSchool(next);
      router.push("/");
      return;
    }
    if (!isHome) {
      return;
    }
    setSchool(next);
  }

  async function handleNotificationClick(
    notificationId: number,
    reviewId: number,
    teacherId: number,
    schoolKey: SchoolKey
  ) {
    if (userId && token) {
      await removeNotification(school, token, notificationId);
      setNotifications((prev) => prev.filter((item) => item.id !== notificationId));
    }
    setNotificationsOpen(false);
    setSchool(schoolKey);
    router.push(`/teachers/${teacherId}#review-${reviewId}`);
  }

  return (
    <header className="bg-transparent">
      <div className="mx-auto w-full max-w-5xl px-4 py-4">
        <div
          className="flex flex-col items-center gap-4 md:grid md:grid-cols-[1fr_auto_1fr]"
        >
          <div className="text-center md:text-left">
            <Link
              href="/"
              className="inline-flex items-center text-2xl font-semibold text-slate-900 transition"
            >
              IDS Teachers
            </Link>
          </div>
          <div className="flex justify-center">
            <div
              className="relative grid w-full max-w-sm grid-cols-3 rounded-full p-1.5 glass-panel"
            >
              <span
                className="absolute left-1 top-1 h-[calc(100%-0.5rem)] w-[calc(33.333%-0.25rem)] rounded-full border border-white/40 glass-tint transition-transform duration-300"
                style={{
                  transform: `translateX(${schoolOptions.indexOf(school) * 100}%)`,
                }}
              />
              {schoolOptions.map((option) => (
                <button
                  key={option}
                  className={`relative z-10 rounded-full px-3 py-2 text-sm font-semibold transition-colors duration-300 ${
                    school === option ? "text-white" : "text-slate-700 hover:text-slate-900"
                  }`}
                  onClick={() => handleSchoolClick(option)}
                  type="button"
                >
                  {schoolThemes[option].label}
                </button>
              ))}
            </div>
          </div>
          <div className="flex justify-center md:justify-end">
            <div className="flex items-center gap-3">
              {token && userId && (
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setNotificationsOpen((open) => !open)}
                    className="flex h-10 w-10 items-center justify-center rounded-full glass-panel text-slate-600 hover:text-slate-900"
                    aria-label="Notifications"
                  >
                    {hasNotifications ? (
                      <BellDot className="h-5 w-5" />
                    ) : (
                      <Bell className="h-5 w-5" />
                    )}
                  </button>
                  {notificationsOpen && (
                    <div className="absolute right-0 z-30 mt-2 w-72 animate-notification-pop rounded-2xl border border-white/60 p-3 text-sm text-slate-700 glass-panel">
                      {notifications.length === 0 ? (
                        <p className="text-sm text-slate-600">no new notifications</p>
                      ) : (
                        <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                          {notifications.map((notification) => (
                            <button
                              key={notification.id}
                              type="button"
                              onClick={() =>
                                handleNotificationClick(
                                  notification.id,
                                  notification.review_id,
                                  notification.teacher_id,
                                  notification.school as SchoolKey
                                )
                              }
                              className="w-full rounded-lg border border-white/50 bg-white/40 px-3 py-2 text-left text-sm text-slate-700 transition hover:text-slate-900"
                            >
                              {notification.type === "upvote" ? "Upvote" : "Downvote"} on your
                              comment
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
              {token && username ? (
                <div
                  className={`flex items-center gap-3 rounded-full px-4 py-2 text-sm font-semibold glass-panel glass-tint-soft ${
                    authSwap ? "animate-auth-swap" : ""
                  }`}
                >
                  <span className="text-theme-primary">{username}</span>
                  <button
                    type="button"
                    onClick={handleLogout}
                    className="text-slate-700 hover:text-slate-900 hover:underline"
                  >
                    Logout
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setAuthOpen(true)}
                  className={`rounded-full border border-white/40 px-4 py-2 text-sm font-semibold text-white shadow-md transition glass-tint backdrop-blur-xl hover:opacity-90 ${
                    authSwap ? "animate-auth-swap" : ""
                  }`}
                >
                  Log in
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
      <AuthPanel open={authOpen} onClose={() => setAuthOpen(false)} />
    </header>
  );
}
