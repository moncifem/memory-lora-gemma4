import { NextRequest, NextResponse } from "next/server";
import { readStatus } from "@/lib/jobs";

export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const status = readStatus(jobId);
  if (!status) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json(status);
}
