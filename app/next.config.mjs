/** @type {import('next').NextConfig} */
const nextConfig = {
  // The /v1/* route handlers stream from the local inference server; disable
  // response buffering by keeping them as dynamic route handlers (default).
  experimental: {
    // Allow long-running pipeline spawns from server actions / route handlers.
  },
};

export default nextConfig;
