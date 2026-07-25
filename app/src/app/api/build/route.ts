import { NextRequest, NextResponse } from "next/server";
import { createJob, listJobs } from "@/lib/jobs";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let body: { repoUrl?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid json" }, { status: 400 });
  }
  const repoUrl = (body.repoUrl || "").trim();
  if (!/^https?:\/\/.+/.test(repoUrl) && !/^git@/.test(repoUrl)) {
    return NextResponse.json({ error: "repoUrl must be a git URL" }, { status: 400 });
  }
  const { jobId, port } = createJob(repoUrl);
  return NextResponse.json({ jobId, port });
}

export async function GET() {
  return NextResponse.json({ jobs: listJobs() });
}
