export type SchoolKey = "sky" | "iss" | "lib";

export interface Teacher {
  id: number;
  name: string;
  subject: string;
  department: string;
  school_id: number;
  bio: string;
}

export interface Review {
  id: number;
  user_id: number;
  teacher_id: number;
  school_id: number;
  rating: number;
  difficulty: number;
  clarity: number;
  comment: string;
  created_at: string;
}

export interface TeacherDetail {
  teacher: Teacher;
  average_rating: number | null;
  reviews: Review[];
}

export interface ModerationReview {
  id: number;
  teacher_id: number;
  school_id: number;
  rating: number;
  difficulty: number;
  clarity: number;
  comment: string;
  created_at: string;
  teacher_name: string;
  user_email: string;
}

export interface UserProfile {
  id: number;
  username: string;
  email: string;
  is_moderator: boolean;
  created_at: string;
}

export interface NotificationItem {
  id: number;
  review_id: number;
  teacher_id: number;
  school: string;
  type: "upvote" | "downvote";
  created_at: string;
}
