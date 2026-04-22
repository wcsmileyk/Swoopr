from django.contrib import admin

from .models import (
    DropzoneAuthorization,
    InstructorPersonalLimits,
    InstructorQualificationEarned,
    InstructorRating,
    PendingInstructorRequest,
    ProgramJump,
    RatingQualification,
    StudentEnrollment,
    StudentJump,
    StudentSignoff,
    USPACategory,
)


class InstructorPersonalLimitsInline(admin.StackedInline):
    model = InstructorPersonalLimits
    extra = 0


class InstructorQualificationEarnedInline(admin.TabularInline):
    model = InstructorQualificationEarned
    extra = 0
    autocomplete_fields = ('qualification', 'dropzone')


@admin.register(InstructorRating)
class InstructorRatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'rating_type', 'earned_date', 'is_current')
    list_filter = ('rating_type',)
    search_fields = ('user__username', 'user__first_name', 'user__last_name')
    ordering = ('user', 'rating_type')
    inlines = [InstructorPersonalLimitsInline, InstructorQualificationEarnedInline]

    @admin.display(boolean=True, description='Current')
    def is_current(self, obj):
        return obj.is_current


@admin.register(RatingQualification)
class RatingQualificationAdmin(admin.ModelAdmin):
    list_display = ('rating_type', 'name', 'portable', 'description')
    list_filter = ('rating_type', 'portable')
    search_fields = ('name',)
    ordering = ('rating_type', 'name')


@admin.register(DropzoneAuthorization)
class DropzoneAuthorizationAdmin(admin.ModelAdmin):
    list_display = ('user', 'dropzone', 'authorization_type', 'stage', 'active', 'cleared_date')
    list_filter = ('authorization_type', 'stage', 'active', 'dropzone')
    search_fields = ('user__username', 'user__first_name', 'user__last_name')
    ordering = ('dropzone', 'authorization_type', 'user')
    autocomplete_fields = ('dropzone', 'instructor_rating')


@admin.register(USPACategory)
class USPACategoryAdmin(admin.ModelAdmin):
    list_display = ('program_type', 'order', 'code', 'name')
    list_filter = ('program_type',)
    ordering = ('program_type', 'order')
    readonly_fields = ('program_type', 'code', 'name', 'order', 'criteria')


class ProgramJumpInline(admin.TabularInline):
    model = ProgramJump
    extra = 0
    fields = ('category', 'jump_number', 'name', 'allowed_methods', 'is_active')


@admin.register(ProgramJump)
class ProgramJumpAdmin(admin.ModelAdmin):
    list_display = ('dropzone', 'category', 'jump_number', 'name', 'is_active')
    list_filter = ('dropzone', 'category__program_type', 'is_active')
    ordering = ('dropzone', 'category__order', 'jump_number')


class StudentJumpInline(admin.TabularInline):
    model = StudentJump
    extra = 0
    fields = ('program_jump', 'jump_method', 'jump_date', 'outcome', 'instructor', 'signed_off_by')
    readonly_fields = ('signed_off_by', 'signed_off_at')


@admin.register(StudentEnrollment)
class StudentEnrollmentAdmin(admin.ModelAdmin):
    list_display = ('student', 'dropzone', 'program_type', 'status', 'current_jump', 'enrolled_date')
    list_filter = ('program_type', 'status', 'dropzone')
    search_fields = ('student__username', 'student__first_name', 'student__last_name')
    ordering = ('-enrolled_date',)
    inlines = [StudentJumpInline]


@admin.register(StudentJump)
class StudentJumpAdmin(admin.ModelAdmin):
    list_display = ('enrollment', 'program_jump', 'jump_method', 'jump_date', 'outcome', 'instructor')
    list_filter = ('outcome', 'jump_method', 'program_jump__category__program_type')
    search_fields = ('enrollment__student__username', 'instructor__username')
    ordering = ('-jump_date',)


@admin.register(StudentSignoff)
class StudentSignoffAdmin(admin.ModelAdmin):
    list_display = ('jump', 'signed_by', 'instructor_name', 'instructor_rating', 'signed_at', 'criteria_modified')
    list_filter = ('instructor_rating', 'criteria_modified', 'student_notified')
    search_fields = ('instructor_name', 'jump__user__username')
    ordering = ('-signed_at',)


@admin.register(PendingInstructorRequest)
class PendingInstructorRequestAdmin(admin.ModelAdmin):
    list_display = ('jump', 'instructor', 'created_at')
    search_fields = ('instructor__username', 'jump__user__username')
    ordering = ('-created_at',)
