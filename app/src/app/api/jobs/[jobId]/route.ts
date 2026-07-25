import { NextRequest, NextResponse } from "next/server";
import { readStatus } from "@/lib/jobs";
import { rmSync, existsSync } from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";

const WORKSPACES_DIR = path.join(process.cwd(), ".workspaces");

/** Delete a build: stop its inference server, then remove its workspace. */
export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  // Reject anything that could escape the workspaces directory.
  if (!/^[a-f0-9]{6,32}$/i.test(jobId)) {
    return NextResponse.json({ error: "invalid job id" }, { status: 400 });
  }
  const dir = path.join(WORKSPACES_DIR, jobId);
  if (!existsSync(dir)) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  // Free the ~10GB the server holds before removing the adapter it loaded.
  const status = readStatus(jobId);
  const pid = status?.server?.pid;
  if (pid) {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
  try {
    rmSync(dir, { recursive: true, force: true });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
  return NextResponse.json({ ok: true, jobId });
}
