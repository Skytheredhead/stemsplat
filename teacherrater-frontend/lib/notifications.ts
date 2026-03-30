import { apiFetch, ensureCsrfToken } from "@/lib/api";
import { NotificationItem, SchoolKey } from "@/lib/types";

export async function loadNotifications(
  school: SchoolKey,
  token: string
): Promise<NotificationItem[]> {
  return apiFetch<NotificationItem[]>("/notifications", { school, token });
}

export async function addNotification(
  school: SchoolKey,
  token: string,
  payload: { review_id: number; type: "upvote" | "downvote" }
): Promise<NotificationItem> {
  await ensureCsrfToken(school);
  return apiFetch<NotificationItem>("/notifications", {
    method: "POST",
    school,
    token,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function removeNotification(
  school: SchoolKey,
  token: string,
  notificationId: number
): Promise<void> {
  await ensureCsrfToken(school);
  await apiFetch(`/notifications/${notificationId}`, {
    method: "DELETE",
    school,
    token,
  });
}
