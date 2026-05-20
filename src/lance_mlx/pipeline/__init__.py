"""Lance task pipelines.

Each task gets its own module with a `<Task>Pipeline` class following the
`from_pretrained` + `__call__` pattern. Wired in during Phases 2-4:

    pipeline/
        understanding.py    x2t_image, x2t_video  (Phase 2)
        t2i.py              text-to-image          (Phase 3)
        image_edit.py       instruction-based edit (Phase 3)
        t2v.py              text-to-video          (Phase 4)
        video_edit.py       video edit             (Phase 4)
"""

# Pipelines are imported lazily on first use to avoid loading mlx-video
# (and its torch dependencies) when only running understanding tasks.

__all__: list[str] = []
