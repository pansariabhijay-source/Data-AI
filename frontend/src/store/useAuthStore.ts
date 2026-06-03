import { create } from "zustand";
import { persist } from "zustand/middleware";
import { authLogout } from "@/lib/api";
import { AUTH_COOKIE, setCookie, deleteCookie } from "@/lib/cookies";

export interface User {
  id: number;
  email: string;
  username: string;
  created_at: string;
  last_login: string | null;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  setAuth: (user: User, token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      setAuth: (user, token) => {
        // Mirror the session into a cookie so the server-side proxy gate can see
        // it on the very next navigation (localStorage is invisible to it).
        setCookie(AUTH_COOKIE, token);
        set({ user, token, isAuthenticated: true });
      },

      logout: () => {
        const { token } = get();
        if (token) {
          // Fire and forget logout on backend
          authLogout(token).catch(console.error);
        }
        deleteCookie(AUTH_COOKIE);
        set({ user: null, token: null, isAuthenticated: false });
      },
    }),
    {
      name: "axiom-auth-store",
      // After the persisted state rehydrates on the client, reconcile the cookie
      // mirror with it — covers the case where the cookie expired or was cleared
      // but the localStorage session is still valid (and vice versa).
      onRehydrateStorage: () => (state) => {
        if (state?.token) {
          setCookie(AUTH_COOKIE, state.token);
        } else {
          deleteCookie(AUTH_COOKIE);
        }
      },
    }
  )
);
