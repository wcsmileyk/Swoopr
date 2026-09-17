from django.contrib import admin

from courses.models import Course, CourseAuditEvent, CourseImport, CourseRevision, CourseSet
from courses.services import archive_course_set, set_visibility


class CourseInline(admin.TabularInline):
    model = Course
    extra = 0
    readonly_fields = ['discipline', 'current_revision', 'archived', 'created_at']
    can_delete = False


@admin.register(CourseSet)
class CourseSetAdmin(admin.ModelAdmin):
    """Global course management (design doc R3) -- requires the
    courses.manage_all_courses permission, not just is_staff, to moderate
    someone else's courses."""

    list_display = ['name', 'owner', 'site_name', 'visibility', 'archived', 'created_at']
    list_filter = ['visibility', 'archived']
    search_fields = ['name', 'owner__username', 'site_name']
    readonly_fields = ['id', 'created_at', 'updated_at']
    inlines = [CourseInline]
    actions = ['make_public', 'make_private', 'archive_sets']

    def has_module_permission(self, request):
        return request.user.has_perm('courses.manage_all_courses')

    has_view_permission = has_change_permission = has_delete_permission = has_module_permission

    def has_add_permission(self, request):
        return False  # course sets are created through the import wizard, not admin

    @admin.action(description='Make selected course sets public')
    def make_public(self, request, queryset):
        for cs in queryset:
            set_visibility(cs, 'public', actor=request.user)

    @admin.action(description='Make selected course sets private')
    def make_private(self, request, queryset):
        for cs in queryset:
            set_visibility(cs, 'private', actor=request.user)

    @admin.action(description='Archive selected course sets')
    def archive_sets(self, request, queryset):
        for cs in queryset:
            archive_course_set(cs, actor=request.user)


@admin.register(CourseRevision)
class CourseRevisionAdmin(admin.ModelAdmin):
    """Revisions are immutable -- read-only in admin, never edited in place."""

    list_display = ['course', 'revision_number', 'source', 'algorithm_version', 'created_by', 'created_at']
    list_filter = ['source', 'algorithm_version']
    readonly_fields = [f.name for f in CourseRevision._meta.fields]

    def has_module_permission(self, request):
        return request.user.has_perm('courses.manage_all_courses')

    has_view_permission = has_module_permission

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(CourseImport)
class CourseImportAdmin(admin.ModelAdmin):
    list_display = ['id', 'owner', 'format', 'input_mode', 'status', 'created_at']
    list_filter = ['format', 'input_mode', 'status']
    readonly_fields = ['id', 'owner', 'source_file', 'file_hash', 'format', 'input_mode',
                       'validation_report', 'extracted', 'committed_course_set', 'created_at']

    def has_module_permission(self, request):
        return request.user.has_perm('courses.manage_all_courses')

    has_view_permission = has_module_permission

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(CourseAuditEvent)
class CourseAuditEventAdmin(admin.ModelAdmin):
    list_display = ['action', 'actor', 'course_set', 'created_at']
    list_filter = ['action']
    readonly_fields = [f.name for f in CourseAuditEvent._meta.fields]

    def has_module_permission(self, request):
        return request.user.has_perm('courses.manage_all_courses')

    has_view_permission = has_module_permission

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
