from omni2.jobs.job import Job, JobSpec, fingerprint, job_id


def make_job(tmp_path, name="clip.wav"):
    src = tmp_path / name
    if not src.exists():
        src.write_bytes(b"fake")
    return Job(JobSpec(input=str(src)), tmp_path / "jobs")


def test_job_id_changes_with_content(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"1")
    first = job_id(str(src))
    src.write_bytes(b"22")
    assert job_id(str(src)) != first


def test_stage_state_persists(tmp_path):
    job = make_job(tmp_path)
    job.set_stage("prepare", "fp1", "done", duration=3.0)
    again = make_job(tmp_path)
    assert again.stage_done("prepare", "fp1")
    assert not again.stage_done("prepare", "fp2")


def test_translation_checkpoint_resume_and_invalidation(tmp_path):
    job = make_job(tmp_path)
    assert job.load_translations("A") == {}
    job.append_translations({0: "一", 1: "二"})
    job.append_translations({2: "三"})
    assert make_job(tmp_path).load_translations("A") == {0: "一", 1: "二", 2: "三"}
    # 設定改過（指紋不同）：舊的翻譯作廢
    assert make_job(tmp_path).load_translations("B") == {}
    assert make_job(tmp_path).load_translations("B") == {}


def test_truncated_last_line_is_ignored(tmp_path):
    job = make_job(tmp_path)
    job.load_translations("A")
    job.append_translations({0: "一"})
    with open(job.translation_path, "a", encoding="utf-8") as f:
        f.write('{"i": 1, "t": "寫到一')
    assert make_job(tmp_path).load_translations("A") == {0: "一"}


def test_fingerprint_is_order_independent_for_dicts():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
